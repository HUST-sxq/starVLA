# /root/starVLA/starVLA/training/trainer_gdpo.py
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
#
# NOTE:
# - This file is designed to be copy-paste runnable WITHOUT depending on VLATrainer in train_starvla.py
# - It implements prepare_training/train/checkpoint/logging loop internally.
# - It uses a practical "Best-of-K" offline GDPO-style update:
#     1) sample K candidate action chunks (no_grad OK)
#     2) compute imitation reward against expert chunk
#     3) pick best action per sample (argmax reward)
#     4) train a differentiable deterministic head output (predict_mu / predict_action) to regress to best action
#   This avoids the non-differentiable sampling issue and will actually update parameters.
#
# - If you later add logp/log_prob into the action head, you can switch to true policy-gradient style.

import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig

logger = get_logger(__name__)


# ----------------------------
# Reward (offline imitation)
# ----------------------------
def reward_fn_action_imitation(
    actions_group: torch.Tensor,         # [B,K,H,A]
    expert_actions: torch.Tensor,        # [B,H,A]
    smooth_lambda: float = 0.0,
    end_weight: float = 2.0,
) -> torch.Tensor:
    """
    Offline imitation reward:
      reward = - weighted_MSE(actions, expert) - smooth_lambda * smoothness_penalty

    Returns:
      reward: [B,K] (higher is better)
    """
    assert actions_group.ndim == 4, f"actions_group must be [B,K,H,A], got {tuple(actions_group.shape)}"
    assert expert_actions.ndim == 3, f"expert_actions must be [B,H,A], got {tuple(expert_actions.shape)}"

    B, K, H, A = actions_group.shape
    w = torch.linspace(1.0, float(end_weight), H, device=actions_group.device).view(1, 1, H, 1)

    diff = actions_group - expert_actions[:, None, :, :]
    mse = ((diff * diff) * w).mean(dim=(-1, -2))  # [B,K]

    if smooth_lambda and smooth_lambda > 0:
        da = actions_group[:, :, 1:, :] - actions_group[:, :, :-1, :]
        smooth = (da * da).mean(dim=(-1, -2))      # [B,K]
    else:
        smooth = 0.0

    return -mse - float(smooth_lambda) * smooth


class GDPOTrailer(TrainerUtils):
    """
    GDPO trainer with a full training loop (so train_starvla.py can call prepare_training()/train()).
    """

    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # will be set in _init_checkpointing()
        self.checkpoint_dir = None
        self.resume_from_checkpoint = None

    # ----------------------------
    # Basic utilities (same style as VLATrainer)
    # ----------------------------
    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        # freeze parameters (reuse TrainerUtils method)
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # distributed prepare
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be first
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _adjust_lr_scheduler_for_resume(self):
        if self.completed_steps > 0:
            logger.info(f"[GDPO] Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(f"[GDPO] LR scheduler now at step {self.completed_steps}, LR={self.lr_scheduler.get_last_lr()}")

    def _get_latest_checkpoint(self, ckpt_dir: str):
        """
        Find latest steps_xxx_pytorch_model.pt under ckpt_dir.
        Returns (path_to_pt, step_int)
        """
        if not os.path.isdir(ckpt_dir):
            return None, 0
        pts = []
        for fn in os.listdir(ckpt_dir):
            if fn.endswith("_pytorch_model.pt") and fn.startswith("steps_"):
                m = re.search(r"steps_(\d+)_pytorch_model\.pt", fn)
                if m:
                    pts.append((int(m.group(1)), os.path.join(ckpt_dir, fn)))
        if not pts:
            return None, 0
        pts.sort(key=lambda x: x[0])
        return pts[-1][1], pts[-1][0]

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, step = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                self.completed_steps = int(step)
                logger.info(f"[GDPO] Resuming from {self.resume_from_checkpoint}, steps={self.completed_steps}")
                return

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            try:
                self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1))
            except Exception:
                self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"[GDPO] Loaded pretrained checkpoint: {pretrained_checkpoint}, steps={self.completed_steps}")
        else:
            logger.info("[GDPO] No pretrained checkpoint provided. Starting from scratch.")
            self.completed_steps = 0

    def _save_checkpoint(self):
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")

            self.accelerator.print(f"✅ [GDPO] Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 [GDPO] Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ [GDPO] Configuration saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics: Dict[str, Any]):
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                metrics["epoch"] = round(self.completed_steps / max(len(self.vla_train_dataloader), 1), 2)
                wandb.log(metrics, step=self.completed_steps)
                logger.info(f"[GDPO] Step {self.completed_steps}, Metrics: {metrics}")

    def _create_data_iterators(self):
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
        return batch_vla

    def _finalize_training(self):
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"[GDPO] Training complete. Final model saved at {final_checkpoint}")
            wandb.finish()
        self.accelerator.wait_for_everyone()

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            logger.info("***** [GDPO] Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    # ----------------------------
    # GDPO core: best-of-K regression
    # ----------------------------
    def _extract_expert_actions(self, batch_vla) -> torch.Tensor:
        """
        Extract expert action chunk from batch:
        - batch_vla can be dict or list[dict]
        - assumes batch contains already-normalized actions (same space as model outputs)
        - uses H = future_action_window_size + 1 (same as your BC forward)
        Returns:
          expert_actions: [B,H,A] float32 on accelerator device
        """
        if isinstance(batch_vla, dict):
            actions = batch_vla["action"]
        else:
            actions = [ex["action"] for ex in batch_vla]

        actions = torch.tensor(np.array(actions), device=self.accelerator.device, dtype=torch.float32)
        H = int(self.config.framework.action_model.future_action_window_size) + 1
        expert_actions = actions[:, -H:, :]
        return expert_actions

    def _predict_deterministic_actions(self, batch_vla) -> torch.Tensor:
        """
        Get a differentiable deterministic prediction from action head.
        Priority:
          1) action_model.predict_mu(vl_embs, state, ...)
          2) action_model.predict_action(vl_embs, state)  (must be deterministic; if it's sampling, this isn't ideal)
          3) fallback: raise error

        Returns:
          pred: [B,H,A] float32, requires grad wrt action head params
        """
        # We want vl_embs (can be detached) and state from the helper you already added.
        # IMPORTANT: pack generation may run under no_grad inside the model helper;
        # that is fine because we do not need gradients w.r.t vl_embs, only w.r.t action head.
        pack = self.model.sample_action_group_for_gdpo(
            examples=batch_vla,
            num_samples=1,   # not used for deterministic, but helper returns vl_embs/state consistently
            num_steps=getattr(getattr(self.config.trainer, "gdpo", None), "mu_steps", None),
            mu_steps=getattr(getattr(self.config.trainer, "gdpo", None), "mu_steps", None),
        )
        vl_embs = pack["vl_embs"]          # [B,L,C] float32
        state = pack["state"]              # [B,1,S] float32 or None

        # Ensure float32 for stability
        if vl_embs.dtype != torch.float32:
            vl_embs = vl_embs.float()
        if state is not None and state.dtype != torch.float32:
            state = state.float()

        # Deterministic head output (differentiable w.r.t head params)
        if hasattr(self.model.action_model, "predict_mu"):
            pred = self.model.action_model.predict_mu(
                vl_embs=vl_embs,
                state=state,
                num_steps=getattr(getattr(self.config.trainer, "gdpo", None), "mu_steps", None),
                z_mode=getattr(getattr(self.config.trainer, "gdpo", None), "z_mode", "zeros"),
                fixed_seed=int(getattr(getattr(self.config.trainer, "gdpo", None), "fixed_seed", 0)),
            )
        elif hasattr(self.model.action_model, "predict_action"):
            # Warning: if predict_action internally samples, this might be stochastic.
            pred = self.model.action_model.predict_action(vl_embs, state)
        else:
            raise RuntimeError(
                "[GDPO] action_model must implement predict_mu(vl_embs, state, ...) "
                "or predict_action(vl_embs, state)."
            )

        # pred should be [B,H,A]
        if pred.ndim == 4:
            # some impl may return [B,1,H,A]
            pred = pred[:, 0]
        pred = pred.to(dtype=torch.float32)
        return pred

    def _sample_action_group(self, batch_vla, K: int, num_steps: Optional[int]) -> Dict[str, Any]:
        """
        Use your model helper to sample K actions:
          returns pack with actions_group [B,K,H,A], vl_embs, state, final_instructions
        """
        pack = self.model.sample_action_group_for_gdpo(
            examples=batch_vla,
            num_samples=int(K),
            num_steps=num_steps,
            mu_steps=getattr(getattr(self.config.trainer, "gdpo", None), "mu_steps", None),
            z_mode=getattr(getattr(self.config.trainer, "gdpo", None), "z_mode", "zeros"),
            fixed_seed=int(getattr(getattr(self.config.trainer, "gdpo", None), "fixed_seed", 0)),
        )
        actions_group = pack["actions_group"]
        if actions_group.dtype != torch.float32:
            actions_group = actions_group.float()
            pack["actions_group"] = actions_group
        return pack

    def _train_step(self, batch_vla):
        """
        Best-of-K offline GDPO-style update (stable, differentiable).
        """
        gdpo_cfg = getattr(self.config.trainer, "gdpo", None)
        K = int(getattr(gdpo_cfg, "num_samples", 4))
        num_steps = getattr(gdpo_cfg, "num_steps", None)   # diffusion steps for sampling group
        smooth_lambda = float(getattr(gdpo_cfg, "smooth_lambda", 0.0))
        end_weight = float(getattr(gdpo_cfg, "end_weight", 2.0))
        beta = float(getattr(gdpo_cfg, "beta", 1.0))       # optional scaling

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # 1) expert chunk
            expert_actions = self._extract_expert_actions(batch_vla)   # [B,H,A] float32

            # 2) sample K candidates (sampling is typically no_grad; OK)
            pack = self._sample_action_group(batch_vla, K=K, num_steps=num_steps)
            actions_group = pack["actions_group"]   # [B,K,H,A] float32

            # 3) compute reward and select best per sample
            reward = reward_fn_action_imitation(
                actions_group=actions_group,
                expert_actions=expert_actions,
                smooth_lambda=smooth_lambda,
                end_weight=end_weight,
            )  # [B,K]

            # best index per batch element
            B = reward.shape[0]
            best_idx = torch.argmax(reward, dim=1)  # [B]
            arange_B = torch.arange(B, device=reward.device)
            best_actions = actions_group[arange_B, best_idx]  # [B,H,A] float32

            # 4) deterministic differentiable prediction (trainable)
            #    NOTE: even if vl_embs are detached, gradients flow into action head parameters.
            pred = self._predict_deterministic_actions(batch_vla)       # [B,H,A] float32, requires_grad

            # 5) regression loss (you can switch to Huber if you want)
            loss_mse = torch.mean((pred - best_actions) ** 2)
            loss = beta * loss_mse

            self.accelerator.backward(loss)

            if getattr(self.config.trainer, "gradient_clipping", None) is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

        # stats
        with torch.no_grad():
            r_mean = float(reward.mean().detach().item())
            r_std = float(reward.std().detach().item())
            r_max = float(reward.max().detach().item())
            r_min = float(reward.min().detach().item())

        return {
            "gdpo_loss": float(loss.detach().item()),
            "gdpo_mse": float(loss_mse.detach().item()),
            "reward_mean": r_mean,
            "reward_std": r_std,
            "reward_max": r_max,
            "reward_min": r_min,
            "K": int(K),
        }

    # ----------------------------
    # Full training loop
    # ----------------------------
    def train(self):
        self._log_training_config()
        self._create_data_iterators()

        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                postfix = {
                    "data_t": f"{(t_end_data - t_start_data):.3f}",
                    "model_t": f"{(t_end_model - t_start_model):.3f}",
                }
                if "gdpo_loss" in step_metrics:
                    postfix["loss"] = f"{step_metrics['gdpo_loss']:.6f}"
                if "reward_mean" in step_metrics:
                    postfix["r"] = f"{step_metrics['reward_mean']:.3f}"
                progress_bar.set_postfix(postfix)

            # add timing
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model

            # log
            self._log_metrics(step_metrics)

            # save
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()
