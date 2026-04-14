import ast
import os
import pathlib

import torch
from monkey_patch_forward import (replace_qwen2_5_with_mixed_modality_forward,
                                  replace_qwen_2_with_mixed_modality_forward)
from transformers import (AutoProcessor, HfArgumentParser,
                          Qwen2_5_VLForConditionalGeneration,
                          Qwen2VLForConditionalGeneration)

from src.dataset import make_grpo_data_module
from src.params import DataArguments, GRPOArguments, ModelArguments
from src.trainer import QwenGRPOTrainer
from src.utils import load_reward_funcs
from train.train_utils import safe_save_model_for_hf_trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True
):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.use_liger_loss = False
    if "Qwen2.5" in model_args.model_id:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward(use_liger=False)
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward(use_liger=False)

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert (
            not training_args.vision_lora
        ), "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError(
            "If `vision_lora` is True, `freeze_vision_tower` must also be True."
        )

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}

    if "Qwen2.5" in model_args.model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2"
            if not training_args.disable_flash_attn2
            else "sdpa",
            **bnb_model_from_pretrained_args,
        )

    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2"
            if not training_args.disable_flash_attn2
            else "sdpa",
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(
        model_to_configure, training_args, compute_dtype, training_args.device
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    peft_config = None

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    dataset_module = make_grpo_data_module(
        model_id=model_args.model_id, processor=processor, data_args=data_args
    )

    reward_funcs, reward_weights = load_reward_funcs(
        "src.train.reward_funcs", training_args
    )
    training_args.reward_weights = reward_weights

    trainer = QwenGRPOTrainer(
        model=model,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module["eval_dataset"],
        reward_funcs=reward_funcs,
        args=training_args,
        peft_config=peft_config,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    import os

    train()
