import os
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import datasets
import torch
import transformers
from accelerate.utils import (broadcast_object_list, gather, gather_object,
                              is_peft_model, set_seed)
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoProcessor,
                          AutoTokenizer, GenerationConfig, PreTrainedModel,
                          PreTrainedTokenizerBase,
                          Qwen2_5_VLForConditionalGeneration,
                          Qwen2VLForConditionalGeneration, Trainer,
                          TrainerCallback, is_wandb_available)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import (PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME,
                                  is_peft_available)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available
from trl.data_utils import (apply_chat_template, is_conversational,
                            maybe_apply_chat_template)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_deepspeed_available, is_rich_available
from trl.models import (create_reference_model, prepare_deepspeed,
                        unwrap_model_for_generation)
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (generate_model_card, get_comet_experiment_url,
                               pad, print_prompt_completions_sample,
                               selective_log_softmax)

from manipulation_model import (get_next_user_input,
                                get_next_user_input_image_cot)
from src.constants import MULTIMODAL_KEYWORDS
from src.train.train_utils import get_peft_state_non_lora_maybe_zero_3
from vision_process import process_vision_info

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

import torch.distributed as dist


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class RepeatRandomSampler(RepeatSampler):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RepeatRandomSampler is deprecated and will be removed in version 0.18. Use RepeatSampler instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean(
        (tensor - torch.nanmean(tensor, keepdim=True)) ** 2
    )  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size]
            if tensor is not None
            else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class QwenGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )

            if "Qwen2.5" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_init_kwargs,
                )
            else:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_init_kwargs,
                )

            if peft_config is not None:
                print("Applying LoRA...")

                def find_all_linear_names(model, multimodal_keywords):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        # LoRA is not applied to the vision modules
                        if any(
                            mm_keyword in name for mm_keyword in multimodal_keywords
                        ):
                            continue
                        if isinstance(module, cls):
                            lora_module_names.add(name)
                    for m in lora_module_names:  # needed for 16-bit
                        if "embed_tokens" in m:
                            lora_module_names.remove(m)
                    return list(lora_module_names)

                target_modules = find_all_linear_names(
                    model, self.vision_modules_keywords
                )
                peft_config.target_modules = target_modules
                model = get_peft_model(model, peft_config)

        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "PEFT is required to use `peft_config`. Run `pip install peft`."
                )
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            if "Qwen2.5" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_init_kwargs,
                )
            else:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_init_kwargs,
                )
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        if args.reward_llm_judge and args.reward_llm_judge != "None":
            print(f"Loading reward llm model from {args.reward_llm_judge}...")
            self.reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_llm_judge)
            self.reward_model = AutoModelForCausalLM.from_pretrained(
                args.reward_llm_judge, torch_dtype="auto"
            )
        else:
            self.reward_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(
                reward_funcs[i], nn.Module
            ):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(
                    reward_funcs[i].config._name_or_path.split("/")[-1]
                )
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions

        # Datasets
        # self.shuffle_dataset = args.shuffle_dataset
        self.shuffle_dataset = False

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict)
                and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = (
            self.accelerator.num_processes
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )
        num_processes = self.accelerator.num_processes
        effective_batch_size = (
            args.per_device_train_batch_size
            * num_processes
            * args.gradient_accumulation_steps
        )
        possible_values = [
            n_gen
            for n_gen in range(2, effective_batch_size + 1)
            if (effective_batch_size) % n_gen == 0
        ]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {args.per_device_train_batch_size} x "
                f"{args.gradient_accumulation_steps}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            effective_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen
                for n_gen in range(2, effective_batch_size + 1)
                if (effective_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The effective eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current effective eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            pad_token_id=processing_class.pad_token_id,
            eos_token_id=processing_class.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        # Shard the reward llm judge model
        if self.reward_model:
            self.reward_model = prepare_deepspeed(self.reward_model, self.accelerator)

        if args.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(
                        reward_func, self.accelerator
                    )
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True
                    )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch, our dataloader loads an *accumulated* batch
    # (i.e., `per_device_batch_size × gradient_accumulation_steps`). This allows us to generate completions
    # once per optimization step—rather than once per gradient accumulation step—which is significantly more efficient.
    # The only change from the original implementation is multiplying the batch size by `gradient_accumulation_steps`.
    # Thus, `_prepare_inputs` is called with the accumulated batch size, and it handles the splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification.As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size
            * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |     Accum step 0      |     Accum step 1      |
        #                                      |   GPU 0   |   GPU 1   |   GPU 0   |   GPU 1   |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     [0   0   1   1   2   2]  3   3   4   4   5   5    <- Generate for the whole accumulated batch; store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1      0   0   1   1   2   2 [ 3   3   4   4   5   5]   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     [0   0   1   1   2   2]  3   3   4   4   5   5    <- Take the stored generations and use the first slice to compute the loss
        #  num_iterations=2 ▼  1          3      0   0   1   1   2   2 [ 3   3   4   4   5   5]   <- Take the stored generations and use the second slice to compute the loss
        #
        #                      2          4     [6   6   7   7   8   8]  9   9  10  10  11  11    <- Generate for the whole accumulated batch; store the completions; use the first slice to compute the loss
        #                      2          5      6   6   7   7   8   8 [ 9   9  10  10  11  11]   <- ...
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: GRPOConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(
        self, model, input_ids, attention_mask, logits_to_keep=None, **multimodal_inputs
    ):
        # unwrap the model to access the model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multimodal_inputs,
            output_hidden_states=True,
        ).hidden_states[-1]
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[
                :, -logits_to_keep:, :
            ]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps_mt(
        self,
        model,
        input_ids,
        attention_mask,
        keep_mask,
        pad_to=None,
        model_name=None,
        **multimodal_inputs,
    ) -> torch.Tensor:
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, **multimodal_inputs
        ).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        keep_mask = keep_mask[:, 1:].to(dtype=torch.bool)
        att = attention_mask[:, 1:].to(dtype=torch.bool)
        keep_mask = keep_mask & att
        logits = logits / self.temperature
        all_logps = selective_log_softmax(logits, input_ids)
        # print('all_logps', all_logps.shape)
        if keep_mask.sum().item() == 0:
            B = all_logps.size(0)
            if pad_to is None or pad_to == 0:
                return torch.zeros(
                    (B, 0), device=all_logps.device, dtype=all_logps.dtype
                )
            return torch.zeros(
                (B, pad_to), device=all_logps.device, dtype=all_logps.dtype
            )

        kept_rows = []
        max_kept = 0
        for b in range(all_logps.size(0)):
            kept = all_logps[b][keep_mask[b]]
            kept_rows.append(kept)
            if kept.numel() > max_kept:
                max_kept = kept.numel()

        if pad_to is not None and pad_to > max_kept:
            max_kept = pad_to

        out = []
        for kept in kept_rows:
            if kept.numel() < max_kept:
                pad = torch.zeros(
                    (max_kept - kept.numel(),), dtype=kept.dtype, device=kept.device
                )
                kept = torch.cat([pad, kept], dim=0)
            out.append(kept.unsqueeze(0))

        per_token_logps_mt = torch.cat(out, dim=0)
        # print('After call - per_token_logps_mt', per_token_logps_mt.shape)
        return per_token_logps_mt

    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs]

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            prompts, return_video_kwargs=True
        )

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            **video_kwargs,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            if prompt_ids.size(1) > self.max_prompt_length:
                print(f"Truncating {len(prompt_ids)} to {self.max_prompt_length}")
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Regular generation path
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            self.accelerator.wait_for_everyone()
            processed_dict = self._multi_round_generate_and_concat(
                unwrapped_model, inputs
            )
            prompt_completion_ids = processed_dict["prompt_completion_ids_st"]
            input_ids_mt = processed_dict["input_ids_mt"]
            attention_mask_mt = processed_dict["attention_mask_mt"]
            keep_mask_mt = processed_dict["keep_mask_mt"]
            multimodal_inputs_mt = processed_dict["multimodal_inputs_mt"]
            completion_mask_mt = processed_dict["completion_mask_mt"]
            pad_to_kept = processed_dict["pad_to_kept"]

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        multimodal_inputs = {
            k: prompt_inputs[k] if k in prompt_inputs else None
            for k in MULTIMODAL_KEYWORDS
        }

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            old_per_token_logps_mt = None

            with unwrap_model_for_generation(
                self.ref_model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as ref_m:
                ref_m.eval()
                ref_per_token_logps_mt = self._get_per_token_logps_mt(
                    ref_m,
                    input_ids_mt,
                    attention_mask_mt,
                    keep_mask_mt,
                    pad_to_kept,
                    "ref_model",
                    **multimodal_inputs_mt,
                )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt[-1]["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
            )
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c} for p, c in zip(prompts, completions)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [
                        key
                        for key in inputs[0]
                        if key not in ["prompt", "completion", "reasoning"]
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs] for key in keys
                    }

                    if reward_func_name == "reasoning_reward":
                        reasoning_gts = [example["reasoning"] for example in inputs]
                        questions = [example["prompt"] for example in inputs]
                        output_reward_func = reward_func(
                            prompts=prompts,
                            completions=completions,
                            reasoning_gts=reasoning_gts,
                            questions=questions,
                            **reward_kwargs,
                        )

                    elif reward_func_name == "accuracy_reward":
                        questions = [example["prompt"] for example in inputs]
                        output_reward_func = reward_func(
                            prompts=prompts,
                            completions=completions,
                            questions=questions,
                            **reward_kwargs,
                        )

                    else:
                        output_reward_func = reward_func(
                            prompts=prompts, completions=completions, **reward_kwargs
                        )

                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        self.accelerator.wait_for_everyone()
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        if self.loss_type in ["mapo", "mapo_bnpo"]:
            # Shape them as [num_groups, G]
            G = self.num_generations
            rewards_group = rewards.view(-1, G)

            # Group-wise stats
            mu = rewards_group.mean(dim=1, keepdim=True)  # μ
            sigma = rewards_group.std(dim=1, keepdim=True, unbiased=False)  # σ
            eps = torch.finfo(rewards_group.dtype).eps

            # Estimate success ratio p = N/G
            # Prefer the discrete "accuracy_reward" column if present; else use r_i >= μ as a light proxy.
            acc_idx = (
                self.reward_func_names.index("reasoning_reward")
                if ("reasoning_reward" in self.reward_func_names)
                else None
            )
            if acc_idx is not None:
                acc_scores = rewards_per_func[:, acc_idx].view(-1, G)
                # treat >=0.5 as success; clamp/round keeps it robust if accuracy is in {0,1} already
                successes = (acc_scores >= 0.5).sum(dim=1, keepdim=True)
            else:
                successes = (rewards_group >= mu).sum(dim=1, keepdim=True)

            p = successes.to(rewards_group.dtype) / G  # p in [0,1]
            lam = 1.0 - 4.0 * p * (1.0 - p)  # λ(p)
            lam = lam.clamp(0.0, 1.0)

            # Two advantages: z-score and percent-deviation
            z = (rewards_group - mu) / (sigma + eps)  # (r_i - μ)/σ
            apd = (rewards_group - mu) / (mu + eps)  # (r_i - μ)/μ

            # Mixed advantage
            A_mapo = (1.0 - lam) * z + lam * apd  # Eq. (6)

            # Flatten back to [num_groups*G]
            advantages = A_mapo.view(-1)

            # For logging continuity we still compute grouped reward stats (not used to scale MAPO)
            mean_grouped_rewards = rewards_group.mean(dim=1)
            std_grouped_rewards = rewards_group.std(dim=1, unbiased=False)

            # Repeat to match rollout shape (for logs only)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(G, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(G, dim=0)

        else:
            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        )
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_mask.float().max().item()
        )

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(
            agg_completion_mask
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_mask.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps_mt,
            "ref_per_token_logps": ref_per_token_logps_mt,
            "multimodal_inputs": multimodal_inputs,
            "input_ids_mt": input_ids_mt,
            "attention_mask_mt": attention_mask_mt,
            "keep_mask_mt": keep_mask_mt,
            "multimodal_inputs_mt": multimodal_inputs_mt,
            "completion_mask_mt": completion_mask_mt,
            "pad_to_kept": pad_to_kept,
        }

    def _multi_round_generate_and_concat(self, unwrapped_model, batch_inputs):
        """
        Multi-round (mt) generation for a single logical video_qa turn.

        For each sample:
        1) Generate Round-1 from the original prompt.
        2) Iteratively append assistant replies and user follow-up inputs, then reprocess vision and regenerate each round.
        3) When 'FINAL_ANSWER' appears, stop adding new rounds to the kept completion/history for scoring, but continue the loop up to NUM_ROUNDS.
        4) Concatenate kept assistant completion tokens across rounds, removing EOS from non-final kept rounds.
        5) Return stitched single-turn sequences of the form: [Round-1 truncated prompt] + [concatenated kept completion].
        6) Build multi-turn tokenized histories from the full generated conversation and use masks to keep only assistant tokens up to the final kept assistant message.
        """
        import copy

        device = self.accelerator.device
        eos_id = self.processing_class.eos_token_id
        pad_id = getattr(self.processing_class, "pad_token_id", 0)

        # Helper: decode a single sequence of token ids to text
        def _decode(tokens_1d: torch.Tensor) -> str:
            return self.processing_class.batch_decode(
                tokens_1d.unsqueeze(0), skip_special_tokens=True
            )[0]

        batch_size = len(batch_inputs)
        assembled_per_sample = []
        mt_histories = []
        last_kept_assistant_msg_idx_list = []

        # Process each sample in the batch independently, then collate
        for idx in range(batch_size):
            sample = batch_inputs[idx]
            history_msgs_all_rounds = copy.deepcopy(sample["prompt"])
            history_msgs_until_final = copy.deepcopy(sample["prompt"])
            segment_frame_list = sample.get("segment_frame_list", None)
            if segment_frame_list is not None:
                segment_frame_list = {int(k): v for k, v in segment_frame_list.items()}

            # Extract media items from the first user message (kept constant unless you later add modify_media())
            # first_msg = [msg for msg in history_msgs if msg["role"] == "user"][0]
            base_media = []
            if (
                isinstance(history_msgs_all_rounds, list)
                and len(history_msgs_all_rounds) > 0
            ):
                for msg in history_msgs_all_rounds:
                    if msg["role"] == "user":
                        for c in msg["content"]:
                            if isinstance(c, dict) and c.get("type") in (
                                "image",
                                "video",
                            ):
                                base_media.append(copy.deepcopy(c))
                        break

            # Accumulate completion tokens across rounds for this sample.
            rounds_completion_tokens = []
            finished = False
            prompt_prefix_tokens = None  # set after round-1
            rounds_kept = 0
            frame_number = None
            last_kept_assistant_msg_idx = -1

            # ----- Round loop -----
            NUM_ROUNDS = 5  # Video-CoM sets max turns to 5
            for r in range(NUM_ROUNDS):
                rounds_kept += 1
                # Build the *text* for this round from the current history.
                round_example = {"prompt": history_msgs_all_rounds}
                round_text = maybe_apply_chat_template(
                    round_example, self.processing_class
                )["prompt"]
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    [history_msgs_all_rounds], return_video_kwargs=True
                )
                round_inputs = self.processing_class(
                    text=[round_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    **video_kwargs,
                )
                round_inputs = super()._prepare_inputs(round_inputs)
                round_ids = round_inputs["input_ids"]

                if (
                    r == 0
                ):  # Save Round-1 prompt prefix and apply same truncation policy as caller
                    prefix = round_ids[0]
                    prefix = prefix[-self.max_prompt_length :]
                    prompt_prefix_tokens = prefix.to(device)

                # Generate for this round (single sample).
                with torch.no_grad():
                    gen_ids = unwrapped_model.generate(
                        **round_inputs, generation_config=self.generation_config
                    )

                round_prompt_len = round_ids.size(1)
                round_full_seq = gen_ids[0].to(device)
                round_completion_ids = round_full_seq[round_prompt_len:]  # (C_r,)
                # Decode to check for FINAL_ANSWER and manage EOS policy.
                round_text_out = _decode(round_completion_ids)
                # Record the index where this assistant message will be inserted.
                assistant_msg_idx = len(history_msgs_all_rounds)

                found_final = "FINAL_ANSWER" in round_text_out

                next_input, frame_number = get_next_user_input(
                    round_text_out,
                    base_media,
                    frame_number,
                    segment_frame_list,
                    dummy=found_final,
                )

                # If this is not the last contributing round, remove EOS tokens.
                if not finished:
                    if not found_final:  # Drop *all* EOS tokens for non-final rounds.
                        round_completion_ids = round_completion_ids[
                            round_completion_ids != eos_id
                        ]

                    if (
                        round_completion_ids.numel() > 0
                    ):  # Keep completion tokens until found_final.
                        rounds_completion_tokens.append(round_completion_ids)

                    # For the until-final history, we do not add a user follow-up after final.
                    history_msgs_until_final = history_msgs_until_final + [
                        {"role": "assistant", "content": round_text_out}
                    ]
                    if not found_final:
                        history_msgs_until_final = history_msgs_until_final + [
                            {"role": "user", "content": next_input}
                        ]

                    # Update kept-assistant index for masking.
                    last_kept_assistant_msg_idx = assistant_msg_idx
                    if found_final:
                        finished = True

                history_msgs_all_rounds = history_msgs_all_rounds + [
                    {"role": "assistant", "content": round_text_out}
                ]
                if next_input is not None:
                    history_msgs_all_rounds = history_msgs_all_rounds + [
                        {"role": "user", "content": next_input}
                    ]
                # print(next_input)

            # after the round loop, if still -1, fall back to the last assistant message in hist
            if last_kept_assistant_msg_idx == -1:
                assistant_idx = [
                    i
                    for i, m in enumerate(history_msgs_all_rounds)
                    if m.get("role") == "assistant"
                ]
                if assistant_idx:
                    last_kept_assistant_msg_idx = assistant_idx[-1]
                else:
                    last_kept_assistant_msg_idx = 0

            # Concatenate per-sample completions across rounds.
            concatenated_completion = torch.cat(rounds_completion_tokens, dim=0).to(
                device
            )
            # Single-turn stitched tokens: [truncated Round-1 prompt] + [all rounds' completion]
            assembled_seq = torch.cat(
                [prompt_prefix_tokens, concatenated_completion], dim=0
            )
            assembled_per_sample.append(assembled_seq)
            # final_histories.append(history_msgs_until_final)
            mt_histories.append(history_msgs_all_rounds)
            last_kept_assistant_msg_idx_list.append(last_kept_assistant_msg_idx)

        # ----- Pad across the batch to a common length -----
        max_len = (
            max(seq.size(0) for seq in assembled_per_sample)
            if assembled_per_sample
            else 0
        )
        padded_batch = []
        for seq in assembled_per_sample:
            pad_len = max_len - seq.size(0)
            if pad_len > 0:
                pad = torch.full((pad_len,), pad_id, dtype=torch.long, device=device)
                seq = torch.cat([seq, pad], dim=0)
            padded_batch.append(seq.unsqueeze(0))
        prompt_completion_ids = torch.cat(padded_batch, dim=0)  # (B, max_len)

        mt_texts = [
            maybe_apply_chat_template({"prompt": h}, self.processing_class)["prompt"]
            for h in mt_histories
        ]
        image_inputs_mt, video_inputs_mt, video_kwargs_mt = process_vision_info(
            mt_histories, return_video_kwargs=True
        )
        mt_inputs = self.processing_class(
            text=mt_texts,
            images=image_inputs_mt,
            videos=video_inputs_mt,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            **video_kwargs_mt,
        )
        mt_inputs = super()._prepare_inputs(mt_inputs)
        input_ids_mt = mt_inputs["input_ids"]
        attention_mask_mt = mt_inputs["attention_mask"]
        B, Lmt = input_ids_mt.size()
        keep_mask_mt = torch.zeros(
            (B, Lmt), dtype=torch.long, device=input_ids_mt.device
        )

        # Build message-level token spans for all messages, then:
        #  (a) zero attention for messages after the kept assistant message,
        #  (b) set keep_mask only for assistant messages up to the kept one,
        #  (c) drop EOS at the end of non-final kept assistant messages.
        for b in range(B):
            hist = mt_histories[b]
            cum_lens = [0] * (len(hist) + 1)
            cur_hist = []
            for j, _ in enumerate(hist):
                cur_hist.append(hist[j])
                if not any(x.get("role") == "user" for x in cur_hist):
                    continue
                sub_text = maybe_apply_chat_template(
                    {"prompt": cur_hist}, self.processing_class
                )["prompt"]
                sub_imgs, sub_vids, sub_vkw = process_vision_info(
                    [cur_hist], return_video_kwargs=True
                )
                sub_inputs = self.processing_class(
                    text=[sub_text],
                    images=sub_imgs,
                    videos=sub_vids,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    **sub_vkw,
                )
                sub_inputs = super()._prepare_inputs(sub_inputs)
                cum_lens[j + 1] = sub_inputs["input_ids"].size(1)

            full_len_single = cum_lens[len(hist)]
            Lmt_b = input_ids_mt[b].size(0)
            offset = Lmt_b - full_len_single

            # Determine masking boundary: everything AFTER this assistant message gets attention=0.
            final_assistant_idx = last_kept_assistant_msg_idx_list[b]
            # Zero attention for all tokens after the final kept assistant message.
            if final_assistant_idx + 1 <= len(hist):
                lo_attn_zero = (
                    offset + cum_lens[min(final_assistant_idx + 1, len(hist))]
                )
                lo_attn_zero = max(0, min(Lmt_b, lo_attn_zero))
                if lo_attn_zero < Lmt_b:
                    attention_mask_mt[b, lo_attn_zero:] = 0

            # Keep only assistant spans up to the final kept assistant.
            assistant_idx = [
                i for i, m in enumerate(hist) if m.get("role") == "assistant"
            ]
            kept_assistant_indices = [
                i for i in assistant_idx if i <= final_assistant_idx
            ]
            for i in kept_assistant_indices:
                lo = offset + cum_lens[i]
                hi = offset + cum_lens[i + 1]
                lo = max(0, min(Lmt_b, lo))
                hi = max(0, min(Lmt_b, hi))
                if hi > lo:
                    keep_mask_mt[b, lo:hi] = 1

            # Drop the trailing EOS for all *non-final* kept assistant messages.
            if len(kept_assistant_indices) > 1:
                for i in kept_assistant_indices[:-1]:
                    hi = offset + cum_lens[i + 1]
                    if 0 < hi <= Lmt_b and input_ids_mt[b, hi - 1].item() == eos_id:
                        keep_mask_mt[b, hi - 1] = 0

        # Build completion_mask_mt length based on kept & attention (right-aligned).
        eff = (keep_mask_mt[:, 1:].bool()) & (attention_mask_mt[:, 1:].bool())
        kept_lens = eff.sum(dim=1)  # shape (B,)
        local_max = kept_lens.max()
        if dist.is_available() and dist.is_initialized():
            world_max = local_max.clone()
            dist.all_reduce(world_max, op=dist.ReduceOp.MAX)
        else:
            world_max = local_max
        pad_to_kept = int(world_max.item())

        if pad_to_kept == 0:
            completion_mask_mt = torch.zeros(
                (B, 0), dtype=torch.long, device=input_ids_mt.device
            )
        else:
            completion_mask_mt = torch.zeros(
                (B, pad_to_kept), dtype=torch.long, device=input_ids_mt.device
            )
            for b in range(B):
                k = int(kept_lens[b].item())
                if k > 0:
                    completion_mask_mt[b, -k:] = 1

        multimodal_inputs_mt = {
            k: mt_inputs[k] for k in MULTIMODAL_KEYWORDS if k in mt_inputs
        }
        self.accelerator.wait_for_everyone()
        return {
            "prompt_completion_ids_st": prompt_completion_ids,
            "input_ids_mt": input_ids_mt,
            "attention_mask_mt": attention_mask_mt,
            "keep_mask_mt": keep_mask_mt,
            "multimodal_inputs_mt": multimodal_inputs_mt,
            "completion_mask_mt": completion_mask_mt,
            "pad_to_kept": pad_to_kept,
        }

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs)
            self._buffered_inputs[
                self._step % self.args.gradient_accumulation_steps
            ] = inputs
        else:
            inputs = self._buffered_inputs[
                self._step % self.args.gradient_accumulation_steps
            ]
        self._step += 1
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        input_ids_mt = inputs["input_ids_mt"]
        attention_mask_mt = inputs["attention_mask_mt"]
        keep_mask_mt = inputs["keep_mask_mt"]
        multimodal_inputs_mt = inputs["multimodal_inputs_mt"]
        completion_mask = inputs["completion_mask_mt"]
        pad_to_kept = inputs["pad_to_kept"]

        per_token_logps = self._get_per_token_logps_mt(
            model,
            input_ids_mt,
            attention_mask_mt,
            keep_mask_mt,
            pad_to_kept,
            "loss_model",
            **multimodal_inputs_mt,
        )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            if ref_per_token_logps.size(1) != per_token_logps.size(1):
                print("Warning: mismatch in the length of the per-token logps!!!!!")
                min_len = min(ref_per_token_logps.size(1), per_token_logps.size(1))
                ref_per_token_logps = ref_per_token_logps[:, :min_len]
                per_token_logps = per_token_logps[:, :min_len]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        # Temp fix if any mismatch in the length of the per-token logps
        old_per_token_logps = (
            per_token_logps.detach()
            if inputs["old_per_token_logps"] is None
            else inputs["old_per_token_logps"]
        )
        if old_per_token_logps.size(1) != per_token_logps.size(1):
            print("Warning: mismatch in the length of the per-token logps!!!!!")
            min_len = min(old_per_token_logps.size(1), per_token_logps.size(1))
            old_per_token_logps = old_per_token_logps[:, :min_len]
            per_token_logps = per_token_logps[:, :min_len]

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        min_len = min(per_token_loss.size(1), completion_mask.size(1))
        per_token_loss = per_token_loss[:, :min_len]
        completion_mask = completion_mask[:, :min_len]

        if self.loss_type in ("grpo", "gmpo", "mapo"):
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type in ("bnpo", "mapo_bnpo"):
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).nanmean().item()
            )

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)]
                    * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def _save_checkpoint(self, model, trial):
        self.accelerator.wait_for_everyone()
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(), require_grad_only=False
            )
            torch.save(
                non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin")
            )

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(QwenGRPOTrainer, self)._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(
            self.model.config._name_or_path
        ):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url()
            if is_wandb_available() and wandb.run is not None
            else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
