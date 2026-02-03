# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T-WorldModel Framework
A lightweight implementation that Qwen-VL + World Model + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from functools import partial


from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
#from starVLA.model.modules.action_model.GR00T_ActionHeader_GDPO import get_action_model, FlowmatchingActionHead
#from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T_WorldModel")
class Qwen_GR00T_WorldModel(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        #self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        
        # ---- choose action head impl (base vs gdpo) ----
        head_impl = str(getattr(self.config.framework.action_model, "head_impl", "base")).lower()

        if head_impl == "gdpo":
            from starVLA.model.modules.action_model.GR00T_ActionHeader_GDPO import get_action_model
        elif head_impl in ["base", "default"]:
            from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
        else:
            raise ValueError(f"Unknown action head_impl={head_impl}. Use base/default/gdpo.")

        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Instruction composition mode:
        #   - "task": use only task instruction
        #   - "subtask": use only subtask instruction (fallback to task if missing)
        #   - "task_subtask": use Task + Subtask together
        #   - "none": ignore all language input (V2A mode)
        # self.instruction_mode = str(getattr(self.config.datasets.vla_data, "instruction_mode", "task")).lower()
        # self.instruction_mode = str(getattr(self.config.datasets.vla_data, "instruction_mode", "task"))
        self.instruction_mode = str(getattr(self.config.datasets.vla_data, "instruction_mode", "task")).lower()


    def _build_instruction(self, task: str, subtask: Optional[str], mode: str) -> str:
        """
        Build the final instruction string for VLM.

        Args:
            task: main task instruction (may be empty)
            subtask: subtask instruction (may be None/empty)
            mode: "task" | "subtask" | "both"
        """
        task = (task or "").strip()
        subtask = (subtask or "").strip()

        if mode == "task":
            return task

        if mode == "subtask":
            # Fallback to task if subtask is missing
            return subtask if subtask else task

        if mode == "task_subtask":
            if subtask:
                return f"Task: {task} | Subtask: {subtask}"
            return task

            # Safe fallback
        return task


    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        if isinstance(examples, dict):
            batch_images = examples["image"]
            actions = examples["action"]
            state = examples.get("state", None)

            # Task instruction (compat: "lang" or "language")
            task_instructions = examples.get("lang", None)
            if task_instructions is None:
                task_instructions = examples.get("language", [""] * len(batch_images))

            # Optional subtask (could be missing)
            subtasks = examples.get("subtask", None)

        else:
            batch_images = [ex["image"] for ex in examples]
            actions = [ex["action"] for ex in examples]
            state = [ex["state"] for ex in examples] if ("state" in examples[0]) else None

            task_instructions = [ex.get("lang", ex.get("language", "")) for ex in examples]
            subtasks = [ex.get("subtask", None) for ex in examples]

        # ---- language switch (V2A mode) ----
        mode = getattr(self, "instruction_mode", "task_subtask")

        if subtasks is None:
            subtasks = [None] * len(task_instructions)

        # Build final instructions based on mode
        if mode == "null":
            final_instructions = [""] * len(batch_images)  # all_instruction to empty string

        elif mode == "task":
            final_instructions = [task_instructions[i] for i in range(len(task_instructions))]

        elif mode == "subtask":
            final_instructions = [
                self._build_instruction(task_instructions[i], subtasks[i], "subtask")
                for i in range(len(task_instructions))
            ]

        elif mode == "task_subtask":
            final_instructions = [
                self._build_instruction(task_instructions[i], subtasks[i], "task_subtask")
                for i in range(len(task_instructions))
            ]

        else:
            raise ValueError(f"Unsupported instruction_mode={mode}. Supported modes: 'task', 'subtask', 'task_subtask', 'null'.")

        #print(f"  Instructions: {final_instructions}")
        
        # Debug: Print out the contents of the batch
        # print("[Debug] Batch Information:")
        # print(f"  Images: {batch_images}")
        # print(f"  Batch size: {len(batch_images)}")
        # print(f"  Task Instructions: {task_instructions}")
        # print(f"  Subtasks: {subtasks}")
        # print(f"  Instructions: {final_instructions}")
        # print(f"  Actions: {actions}")
        # batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        # #instructions = [example["lang"] for example in examples]   #  [B, str]
        # actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        # state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        # instructions = []
        # for ex in examples:
        #     s = ex.get("lang", "") 
        #     if not self.use_language:
        #         s = self.empty_language  # 通常就是 " "
        #     instructions.append(s)

        # if len(instructions) > 0:
        #     print("[Debug] use_language:", self.use_language, "instr0_len:", len(instructions[0]))
         # Debug: Print out the contents of the batch with the frame number
        # for i, image in enumerate(batch_images):
        #     #print(f"Frame {i + 1}/{len(batch_images)}:")
        #     print(f"  Task Instruction: {task_instructions[i]}")
        #     print(f"  Subtask: {subtasks[i]}")
            #print(f"  Action: {actions[i]}")
            
        # Debug: Print the batch size and instructions
        # print(f"  Batch size: {len(batch_images)}")
        # print(f"  Task Instructions: {task_instructions}")
        # print(f"  Subtasks: {subtasks}")
    


        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=final_instructions)
        #qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 1) cast vl to fp32 (important)
            last_hidden_fp32 = last_hidden.float()

            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=torch.float32
            )  # [B, T_full, action_dim]

            actions_target = actions[:, -(self.future_action_window_size + 1):, :]  # [B, H, A]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)  # fp32
            last_hidden_repeated = last_hidden_fp32.repeat(repeated_diffusion_steps, 1, 1)   # fp32

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=torch.float32
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        #instructions = [example["lang"] for example in examples]  # [B, str]
    
        # Build task/subtask instructions
        task_instructions = [ex.get("lang", ex.get("language", "")) for ex in examples]
        subtasks = [ex.get("subtask", None) for ex in examples]

        # ---- instruction switch: task / subtask / task_subtask / null ----
        mode = getattr(self, "instruction_mode", "task_subtask")

        if mode == "null":
            final_instructions = ["" for _ in range(len(task_instructions))]
        else:
            final_instructions = [
                self._build_instruction(task_instructions[i], subtasks[i], mode)
                for i in range(len(task_instructions))
            ]



        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=final_instructions)
        #qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
    
    @torch.no_grad()
    def sample_action_group_for_gdpo(
        self,
        examples,
        num_samples: int = 4,          # K
        num_steps: int = None,         # diffusion/flow steps for sampling
        mu_steps: int = None,          # steps for mu (can be same as num_steps)
        z_mode: str = "zeros",         # "zeros" or "fixed_rand"
        fixed_seed: int = 0,
    ):
        """
        For GDPO/GRPO post-training:
        - Encode (image, instruction) -> vl_embs
        - Sample K candidate action chunks -> actions_group [B,K,H,A]
        - Get deterministic reference action -> mu [B,H,A] (optional but recommended)

        Returns dict with:
            vl_embs: [B, L, C] float32
            actions_group: [B, K, H, A] float32
            mu: [B, H, A] float32
            state: [B, 1, state_dim] float32 or None
            final_instructions: List[str]  (for debugging / logging)
        """

        # ---- normalize examples to list ----
        if isinstance(examples, dict):
            batch_images = examples["image"]
            state = examples.get("state", None)
            task_instructions = examples.get("lang", None)
            if task_instructions is None:
                task_instructions = examples.get("language", [""] * len(batch_images))
            subtasks = examples.get("subtask", None)
        else:
            batch_images = [ex["image"] for ex in examples]
            state = [ex["state"] for ex in examples] if ("state" in examples[0]) else None
            task_instructions = [ex.get("lang", ex.get("language", "")) for ex in examples]
            subtasks = [ex.get("subtask", None) for ex in examples]

        if subtasks is None:
            subtasks = [None] * len(task_instructions)

        # ---- build final instructions according to instruction_mode ----
        mode = getattr(self, "instruction_mode", "task_subtask")

        if mode == "null":
            final_instructions = [""] * len(batch_images)
        elif mode == "task":
            final_instructions = [task_instructions[i] for i in range(len(task_instructions))]
        elif mode == "subtask":
            final_instructions = [
                self._build_instruction(task_instructions[i], subtasks[i], "subtask")
                for i in range(len(task_instructions))
            ]
        elif mode == "task_subtask":
            final_instructions = [
                self._build_instruction(task_instructions[i], subtasks[i], "task_subtask")
                for i in range(len(task_instructions))
            ]
        else:
            raise ValueError(f"Unsupported instruction_mode={mode}")

        # ---- optional resize (keep consistent with predict_action) ----
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            # if your batch_images are PILs already, resize_images works;
            # if not, make sure your dataloader provides compatible format
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # ---- VLM forward -> vl_embs ----
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=final_instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs = qwenvl_outputs.hidden_states[-1]   # [B, L, C] bf16

        # ---- cast to fp32 for action head stability ----
        vl_embs = vl_embs.float()

        state_tensor = None
        if state is not None:
            state_tensor = torch.tensor(np.array(state), device=vl_embs.device, dtype=torch.float32)

        # ---- require GDPO-capable head ----
        if not hasattr(self.action_model, "sample_actions"):
            raise RuntimeError(
                "Current action_model does NOT support sample_actions(). "
                "Please set config.framework.action_model.head_impl: gdpo"
            )

        # ---- sample group actions ----
        actions_group = self.action_model.sample_actions(
            vl_embs=vl_embs,
            state=state_tensor,
            num_samples=int(num_samples),
            num_steps=num_steps,
        )  # [B, K, H, A]

        # ---- compute deterministic mu (recommended for GDPO) ----
        if hasattr(self.action_model, "predict_mu"):
            mu = self.action_model.predict_mu(
                vl_embs=vl_embs,
                state=state_tensor,
                num_steps=mu_steps if mu_steps is not None else num_steps,
                z_mode=z_mode,
                fixed_seed=int(fixed_seed),
            )  # [B, H, A]
        else:
            mu = None

        return {
            "vl_embs": vl_embs,
            "actions_group": actions_group,
            "mu": mu,
            "state": state_tensor,
            "final_instructions": final_instructions,
        }




if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    #args.config_yaml  = "examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    
    model: Qwen_GR00T_WorldModel = Qwen_GR00T_WorldModel(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        #collate_fn=collate_fn,
        collate_fn=partial(collate_fn, data_cfg=vla_dataset_cfg),
    )
    # forward model with dataloader
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        # break

    action = model.predict_action(examples=batch)
    print("Finished")