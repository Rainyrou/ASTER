"""
2025.8.8
2025.8.7
4.55.0
0.9.6
__UNSLOTH_VERSIONING__
"""
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from trl.trainer.sft_trainer import (AutoModelForCausalLM, AutoTokenizer, Callable, ConstantLengthDataset, DataCollator, DataCollatorForCompletionOnlyLM, DataCollatorForLanguageModeling, Dataset, DatasetGenerationError, Dict, EvalPrediction, List, Optional, PartialState, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, RichProgressCallback, SFTConfig, SFTTrainer, SchemaInferenceError, Trainer, TrainerCallback, Tuple, Union, _deprecate_arguments, dataclasses, datasets, get_formatting_func_from_dataset, get_peft_model, inspect, is_peft_available, neftune_post_forward_hook, nn, peft_module_casting_to_bf16, prepare_model_for_kbit_training, torch, trl_sanitze_kwargs_for_tagging, unwrap_model, warnings, wraps, ConstantLengthDataset, Dataset, datasets, torch, warnings, DataCollator, DataCollatorForLanguageModeling, inspect, warnings)


import os
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(logits, index):
    # Split into 4 chunks only
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = 4, dim = 0)
    chunked_index  = torch.chunk(index.reshape(-1), chunks = 4, dim = 0)
    all_per_token_logps = []
    # Below loop does the same as selective_log_softmax(chunk_logits, chunk_index)
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        selected_logits = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)
    pass
    all_per_token_logps = torch.concat(all_per_token_logps)
    all_per_token_logps = all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))
    return all_per_token_logps
@dataclass
class UnslothSFTConfig(SFTConfig):
    """
    
    Initialize SFTConfig.

    Args:
        dataset_text_field (`Optional[str]`):
            The name of the text field of the dataset, in case this is passed by a user, the trainer will automatically create a
            `ConstantLengthDataset` based on the `dataset_text_field` argument. Defaults to None.
        packing (`Optional[bool]`):
            Used only in case `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences
            of the dataset. Defaults to False.
        max_seq_length (`Optional[int]`):
            The maximum sequence length to use for the `ConstantLengthDataset` and for automatically creating the Dataset. Defaults to min of the smaller of the `tokenizer.model_max_length` and `1024`.
        dataset_num_proc (`Optional[int]`):
            The number of workers to use to tokenize the data. Only used when `packing=False`. Defaults to None.
        dataset_batch_size (`int`):
            The number of examples to tokenize per batch. If batch_size <= 0 or batch_size == None,
            tokenize the full dataset as a single batch. Defaults to 1000.
        neftune_noise_alpha (`Optional[float]`):
            If not `None`, this will activate NEFTune noise embeddings. This has been proven to drastically improve model performances for instruction
            fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string.
        dataset_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when creating packed or non-packed datasets
        eval_packing: (`Optional[bool]`, *optional*):
            Whether to pack the eval dataset as well. Defaults to `packing` if `None` is passed.
        num_of_sequences (`Optional[int]`):
            The number of sequences to use for the `ConstantLengthDataset`. Defaults to `1024`.
        chars_per_token (`Optional[float]`):
            The number of characters per token to use for the `ConstantLengthDataset`. Defaults to `3.6`. You can check how this is computed in the
            stack-llama example:
            [chars_token_ratio](https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53).
    
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    def __init__(
        self,
        output_dir = None,
        overwrite_output_dir = None,
        do_train = False,
        do_eval = False,
        do_predict = False,
        eval_strategy = 'no',
        prediction_loss_only = False,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        per_gpu_train_batch_size = None,
        per_gpu_eval_batch_size = None,
        gradient_accumulation_steps = 2,
        eval_accumulation_steps = 2,
        eval_delay = 0,
        torch_empty_cache_steps = 250,
        learning_rate = 5e-05,
        weight_decay = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        max_grad_norm = 1.0,
        num_train_epochs = 3.0,
        max_steps = -1,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.1,
        warmup_steps = 0,
        log_level = 'passive',
        log_level_replica = 'warning',
        log_on_each_node = True,
        logging_dir = None,
        logging_strategy = 'steps',
        logging_first_step = False,
        logging_steps = 1,
        logging_nan_inf_filter = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_total_limit = None,
        save_safetensors = True,
        save_on_each_node = False,
        save_only_model = False,
        restore_callback_states_from_checkpoint = False,
        no_cuda = False,
        use_cpu = False,
        use_mps_device = False,
        seed = 3407,
        data_seed = 3407,
        jit_mode_eval = False,
        use_ipex = False,
        bf16 = False,
        fp16 = False,
        fp16_opt_level = 'O1',
        half_precision_backend = 'auto',
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        local_rank = -1,
        ddp_backend = None,
        tpu_num_cores = None,
        tpu_metrics_debug = False,
        debug = '',
        dataloader_drop_last = False,
        eval_steps = None,
        dataloader_num_workers = 0,
        dataloader_prefetch_factor = None,
        past_index = -1,
        run_name = None,
        disable_tqdm = None,
        remove_unused_columns = True,
        label_names = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        fsdp = '',
        fsdp_min_num_params = 0,
        fsdp_config = None,
        fsdp_transformer_layer_cls_to_wrap = None,
        accelerator_config = None,
        deepspeed = None,
        label_smoothing_factor = 0.0,
        optim = 'adamw_8bit',
        optim_args = None,
        adafactor = False,
        group_by_length = False,
        length_column_name = 'length',
        report_to = None,
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        skip_memory_metrics = True,
        use_legacy_prediction_loop = False,
        push_to_hub = False,
        resume_from_checkpoint = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_token = None,
        hub_private_repo = None,
        hub_always_push = False,
        hub_revision = None,
        gradient_checkpointing = False,
        gradient_checkpointing_kwargs = None,
        include_inputs_for_metrics = False,
        eval_do_concat_batches = True,
        fp16_backend = 'auto',
        push_to_hub_model_id = None,
        push_to_hub_organization = None,
        push_to_hub_token = None,
        mp_parameters = '',
        auto_find_batch_size = True,
        full_determinism = False,
        torchdynamo = None,
        ray_scope = 'last',
        ddp_timeout = 1800,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        include_tokens_per_second = False,
        include_num_input_tokens_seen = False,
        neftune_noise_alpha = None,
        optim_target_modules = None,
        batch_eval_metrics = False,
        eval_on_start = False,
        use_liger_kernel = False,
        liger_kernel_config = None,
        eval_use_gather_object = False,
        average_tokens_across_devices = True,
        dataset_text_field = None,
        packing = False,
        max_seq_length = None,
        dataset_num_proc = None,
        dataset_batch_size = 1000,
        model_init_kwargs = None,
        dataset_kwargs = None,
        eval_packing = None,
        num_of_sequences = 1024,
        chars_per_token = 3.6,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        **kwargs,
    ):
        if learning_rate < 1e-7: raise FloatingPointError(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: raise OverflowError(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        if dataset_num_proc is None:
            from multiprocessing import cpu_count
            dataset_num_proc = min(cpu_count()*2, 2)
        
        super().__init__(
            output_dir = output_dir,
            overwrite_output_dir = overwrite_output_dir,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            eval_strategy = eval_strategy,
            prediction_loss_only = prediction_loss_only,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            per_gpu_train_batch_size = per_gpu_train_batch_size,
            per_gpu_eval_batch_size = per_gpu_eval_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            eval_accumulation_steps = eval_accumulation_steps,
            eval_delay = eval_delay,
            torch_empty_cache_steps = torch_empty_cache_steps,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            max_grad_norm = max_grad_norm,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            lr_scheduler_type = lr_scheduler_type,
            warmup_ratio = warmup_ratio,
            warmup_steps = warmup_steps,
            log_level = log_level,
            log_level_replica = log_level_replica,
            log_on_each_node = log_on_each_node,
            logging_dir = logging_dir,
            logging_strategy = logging_strategy,
            logging_first_step = logging_first_step,
            logging_steps = logging_steps,
            logging_nan_inf_filter = logging_nan_inf_filter,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            save_safetensors = save_safetensors,
            save_on_each_node = save_on_each_node,
            save_only_model = save_only_model,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            no_cuda = no_cuda,
            use_cpu = use_cpu,
            use_mps_device = use_mps_device,
            seed = seed,
            data_seed = data_seed,
            jit_mode_eval = jit_mode_eval,
            use_ipex = use_ipex,
            bf16 = bf16,
            fp16 = fp16,
            fp16_opt_level = fp16_opt_level,
            half_precision_backend = half_precision_backend,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            local_rank = local_rank,
            ddp_backend = ddp_backend,
            tpu_num_cores = tpu_num_cores,
            tpu_metrics_debug = tpu_metrics_debug,
            debug = debug,
            dataloader_drop_last = dataloader_drop_last,
            eval_steps = eval_steps,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            past_index = past_index,
            run_name = run_name,
            disable_tqdm = disable_tqdm,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            fsdp = fsdp,
            fsdp_min_num_params = fsdp_min_num_params,
            fsdp_config = fsdp_config,
            fsdp_transformer_layer_cls_to_wrap = fsdp_transformer_layer_cls_to_wrap,
            accelerator_config = accelerator_config,
            deepspeed = deepspeed,
            label_smoothing_factor = label_smoothing_factor,
            optim = optim,
            optim_args = optim_args,
            adafactor = adafactor,
            group_by_length = group_by_length,
            length_column_name = length_column_name,
            report_to = report_to,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            skip_memory_metrics = skip_memory_metrics,
            use_legacy_prediction_loop = use_legacy_prediction_loop,
            push_to_hub = push_to_hub,
            resume_from_checkpoint = resume_from_checkpoint,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_always_push = hub_always_push,
            hub_revision = hub_revision,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            include_inputs_for_metrics = include_inputs_for_metrics,
            eval_do_concat_batches = eval_do_concat_batches,
            fp16_backend = fp16_backend,
            push_to_hub_model_id = push_to_hub_model_id,
            push_to_hub_organization = push_to_hub_organization,
            push_to_hub_token = push_to_hub_token,
            mp_parameters = mp_parameters,
            auto_find_batch_size = auto_find_batch_size,
            full_determinism = full_determinism,
            torchdynamo = torchdynamo,
            ray_scope = ray_scope,
            ddp_timeout = ddp_timeout,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            include_tokens_per_second = include_tokens_per_second,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            neftune_noise_alpha = neftune_noise_alpha,
            optim_target_modules = optim_target_modules,
            batch_eval_metrics = batch_eval_metrics,
            eval_on_start = eval_on_start,
            use_liger_kernel = use_liger_kernel,
            liger_kernel_config = liger_kernel_config,
            eval_use_gather_object = eval_use_gather_object,
            average_tokens_across_devices = average_tokens_across_devices,
            dataset_text_field = dataset_text_field,
            packing = packing,
            max_seq_length = max_seq_length,
            dataset_num_proc = dataset_num_proc,
            dataset_batch_size = dataset_batch_size,
            model_init_kwargs = model_init_kwargs,
            dataset_kwargs = dataset_kwargs,
            eval_packing = eval_packing,
            num_of_sequences = num_of_sequences,
            chars_per_token = chars_per_token,**kwargs)
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
pass

class _UnslothSFTTrainer(Trainer):
    r""""""

    _tag_names = ["trl", "sft"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "dataset_text_field",
            "packing",
            "max_seq_length",
            "dataset_num_proc",
            "dataset_batch_size",
            "neftune_noise_alpha",
            "model_init_kwargs",
            "dataset_kwargs",
            "eval_packing",
            "num_of_sequences",
            "chars_per_token",
        ],
        custom_message="Deprecated positional argument(s) used in SFTTrainer, please use the SFTConfig to set these arguments instead.",
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        eval_packing: Optional[bool] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            warnings.warn(f"No `SFTConfig` passed, using `output_dir={output_dir}`.")
            args = SFTConfig(output_dir=output_dir)
        elif args is not None and args.__class__.__name__ == "TrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict[] redacts them
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = SFTConfig(**args_as_dict)

        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.model_init_kwargs = model_init_kwargs
        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs to the SFTConfig, but your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs

            torch_dtype = model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)

                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the SFTConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )

            model_init_kwargs["torch_dtype"] = torch_dtype

        if infinite is not None:
            warnings.warn(
                "The `infinite` argument is deprecated and will be removed in a future version of TRL. Use `TrainingArguments.max_steps` or `TrainingArguments.num_train_epochs` instead to control training length."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if packing:
            warnings.warn(
                "You passed a `packing` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.packing = packing
        if eval_packing is not None:
            warnings.warn(
                "You passed a `eval_packing` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.eval_packing = eval_packing

        if args.packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                is_sharded_qlora = False
                # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
                # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
                # QLoRA + FSDP / DS-Zero3
                if getattr(model, "is_loaded_in_4bit", False):
                    for _, param in model.named_parameters():
                        if param.__class__.__name__ == "Params4bit":
                            is_sharded_qlora = param.data.device.type == "cpu"
                            break
                if getattr(model, "is_loaded_in_8bit", False) or (
                    getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora
                ):
                    prepare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                    if args is not None:
                        args = dataclasses.replace(args, gradient_checkpointing=False)
                elif getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                if (
                    "autocast_adapter_dtype" in list(inspect.signature(get_peft_model).parameters)
                    and getattr(model, "is_loaded_in_4bit", False)
                    and is_sharded_qlora
                ):
                    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
                else:
                    model = model
                if (
                    args is not None
                    and args.bf16
                    and getattr(model, "is_loaded_in_4bit", False)
                    and not is_sharded_qlora
                ):
                    peft_module_casting_to_bf16(model)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_seq_length is not None:
            warnings.warn(
                "You passed a `max_seq_length` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.max_seq_length = max_seq_length

        if args.max_seq_length is None:
            # to overcome some issues with broken tokenizers
            max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed a `dataset_num_proc` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc

        if dataset_batch_size != args.dataset_batch_size:
            warnings.warn(
                "You passed a `dataset_batch_size` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_batch_size = dataset_batch_size
        self.dataset_batch_size = args.dataset_batch_size

        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")
        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            # self.neftune_noise_alpha is done at Trainer level
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if dataset_text_field is not None:
            warnings.warn(
                "You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_text_field = dataset_text_field

        if formatting_func is None and args.dataset_text_field is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays #None
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)
            # if a template is detected, we don't need to add special tokens again
            if formatting_func is not None:
                if dataset_kwargs is None:
                    dataset_kwargs = {"add_special_tokens": False}
                else:
                    dataset_kwargs["add_special_tokens"] = False

        if not args.packing:
            # If we aren't skipping data preparation, then a dataset_text_field
            # or formatting_func must be provided.
            if (
                args.dataset_text_field is None
                and formatting_func is None
                and dataset_kwargs is not None
                and "skip_prepare_dataset" in dataset_kwargs
                and dataset_kwargs["skip_prepare_dataset"]
            ):
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer/SFTConfig, but you didn't pass a `dataset_text_field` or `formatting_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        if num_of_sequences != args.num_of_sequences:
            warnings.warn(
                "You passed a `num_of_sequences` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.num_of_sequences = num_of_sequences

        if chars_per_token != args.chars_per_token:
            warnings.warn(
                "You passed a `chars_per_token` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.chars_per_token = chars_per_token

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if dataset_kwargs is not None:
                warnings.warn(
                    "You passed a `dataset_kwargs` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
                )
                args.dataset_kwargs = dataset_kwargs
            if args.dataset_kwargs is None:
                args.dataset_kwargs = {}
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    tokenizer,
                    args.packing,
                    args.dataset_text_field,
                    args.max_seq_length,
                    formatting_func,
                    args.num_of_sequences,
                    args.chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **args.dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                eval_packing = args.packing if args.eval_packing is None else args.eval_packing

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        tokenizer,
                        eval_packing,
                        args.dataset_text_field,
                        args.max_seq_length,
                        formatting_func,
                        args.num_of_sequences,
                        args.chars_per_token,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **args.dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

        if tokenizer.padding_side is not None and tokenizer.padding_side != "right":
            warnings.warn(
                "You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to "
                "overflow issues when training a model in half-precision. You might consider adding `tokenizer.padding_side = 'right'` to your code."
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.args.max_steps > 0 and args.packing:
            warnings.warn(
                "You passed `packing=True` to the SFTTrainer/SFTConfig, and you are training your model with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and args.packing:
            self.train_dataset.infinite = False

        if any(isinstance(callback, RichProgressCallback) for callback in self.callback_handler.callbacks):
            for callback in self.callback_handler.callbacks:
                # Remove the PrinterCallback to avoid duplicated prints in case we passed a `RichProgressCallback`
                if callback.__class__.__name__ == "PrinterCallback":
                    self.callback_handler.pop_callback(callback)

    @wraps(Trainer.train)
    def train(self, *args, **kwargs):
        # Activate neftune right before training.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            self.model = self._trl_activate_neftune(self.model)

        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            unwrapped_model = unwrap_model(self.model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                embeddings = unwrapped_model.base_model.model.get_input_embeddings()
            else:
                embeddings = unwrapped_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            del embeddings.neftune_noise_alpha

        return output

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "sft" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):
        if 'skip_prepare_dataset' in locals() and skip_prepare_dataset:
            return dataset
        if 'tokenizer'          not in locals(): tokenizer = processing_class
        if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')
        if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field
        if 'dataset_text_field' not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `dataset_text_field` does not exist!')
        test_text = dataset[0][dataset_text_field] if (formatting_func is None and dataset_text_field is not None) else formatting_func(dataset[0])[0]
        chat_template = getattr(tokenizer, 'chat_template', None)
        chat_template = '' if chat_template is None else chat_template
        has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) if getattr(tokenizer, 'bos_token', None) is not None else False
        if 'add_special_tokens' not in locals() and has_bos_token_already:
            from functools import partial
            tokenizer_call = tokenizer.__call__
            tokenizer.__call__ = partial(tokenizer_call, add_special_tokens = False)
            processing_class = tokenizer
        else:
            tokenizer_call = None
            add_special_tokens = False if has_bos_token_already else locals().get('add_special_tokens', False)
        if dataset is None:
            raise ValueError("The dataset should not be None")

        if skip_prepare_dataset:
            return dataset

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored."
                )

            return dataset

        # check if torch dataset / dataloader and do nothing
        # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
        if isinstance(
            dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)
        ) and not isinstance(dataset, datasets.IterableDataset):
            return dataset

        if not packing:
            return self._prepare_non_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                add_special_tokens,
                remove_unused_columns,
            )

        else:
            return self._prepare_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                num_of_sequences,
                chars_per_token,
                formatting_func,
                append_concat_token,
                add_special_tokens,
            )

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        if 'skip_prepare_dataset' in locals() and skip_prepare_dataset:
            return dataset
        if 'tokenizer'          not in locals(): tokenizer = processing_class
        if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')
        if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field
        if 'dataset_text_field' not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `dataset_text_field` does not exist!')
        test_text = dataset[0][dataset_text_field] if (formatting_func is None and dataset_text_field is not None) else formatting_func(dataset[0])[0]
        chat_template = getattr(tokenizer, 'chat_template', None)
        chat_template = '' if chat_template is None else chat_template
        has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) if getattr(tokenizer, 'bos_token', None) is not None else False
        if 'add_special_tokens' not in locals() and has_bos_token_already:
            from functools import partial
            tokenizer_call = tokenizer.__call__
            tokenizer.__call__ = partial(tokenizer_call, add_special_tokens = False)
            processing_class = tokenizer
        else:
            tokenizer_call = None
            add_special_tokens = False if has_bos_token_already else locals().get('add_special_tokens', False)
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )
        if tokenizer_call is not None: tokenizer.__call__ = tokenizer_call
        return tokenized_dataset

    def _prepare_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if dataset_text_field is not None or formatting_func is not None:
            if tokenizer is None:
                raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")

            constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
            )

            if isinstance(dataset, datasets.IterableDataset):
                return constant_length_iterator

            def data_generator(constant_length_iterator):
                yield from constant_length_iterator

            try:
                packed_dataset = Dataset.from_generator(
                    data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
                )
            except (DatasetGenerationError, SchemaInferenceError) as exc:
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence."
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
            )

    def _trl_activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        """
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model
class UnslothSFTTrainer(_UnslothSFTTrainer):
    """
    
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`SFTConfig`]):
            The arguments to tweak for training. Will default to a basic instance of [`SFTConfig`] with the `output_dir`
            set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to None):
            The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values.
            If not specified, only the loss will be computed during evaluation.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
    
    """
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        preprocess_logits_for_metrics = None,
        peft_config = None,
        dataset_text_field = None,
        packing = False,
        formatting_func = None,
        max_seq_length = None,
        infinite = None,
        num_of_sequences = 1024,
        chars_per_token = 3.6,
        dataset_num_proc = None,
        dataset_batch_size = 1000,
        neftune_noise_alpha = None,
        model_init_kwargs = None,
        dataset_kwargs = None,
        eval_packing = None,
        **kwargs
    ):
        if args is None: args = UnslothSFTConfig()
        use_bf16 = getattr(args, 'bf16', False)
        if type(use_bf16) is not bool: use_bf16 = False
        use_fp16 = getattr(args, 'fp16', False)
        if type(use_fp16) is not bool: use_fp16 = False
        force_float32 = False
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1':
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            args.fp16 = float16
            args.bf16 = not float16
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        if type(fp16_full_eval) is not bool: fp16_full_eval = False
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if type(bf16_full_eval) is not bool: bf16_full_eval = False
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        _output_logits = False
        if locals().get('compute_metrics', None) is not None: _output_logits = True
        if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True
        if _output_logits:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
        if 'max_length' not in locals() and not hasattr(args, 'max_length'):
            pass
        else:
            if hasattr(args, 'max_seq_length') and args.max_seq_length is not None and args.max_seq_length > 0:
                if hasattr(args, 'max_length'):
                    args.max_length = args.max_seq_length
                    max_length = args.max_length
            else:
                model_max_length = getattr(model, 'max_seq_length', None)
                # print(model_max_length, 'mml1')
                if model_max_length is None: model_max_length = getattr(model, 'max_length', None)
                # print(model_max_length, 'mml2')
                if model_max_length is not None:
                    args.max_length = model_max_length
                    max_length = args.max_length
                elif hasattr(args, 'max_length') and args.max_length is not None:
                    max_length = args.max_length
                    # if we are here, then we are in a weird case where max_length is set but max_seq_length is not set
                    setattr(model, 'max_seq_length', max_length)
                else:
                    print('Unsloth: We did not find `max_seq_length` or `max_length` in the model or args. We will set it to 1024.')
                    args.max_length = 1024
        if model is not None and hasattr(model, 'for_training'):
            model.for_training()
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        __tokenizer = processing_class if 'processing_class' in locals() else tokenizer
        from unsloth_zoo.vision_utils import UnslothVisionDataCollator
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in train_dataset.column_names:
                data_collator = TransformersDataCollatorForLanguageModeling(__tokenizer, mlm = False, mlm_probability = 0.0)
            elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in train_dataset.column_names:
                data_collator = DataCollatorForSeq2Seq(__tokenizer)
        else:
            if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False
            if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''
            if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):
                if isinstance(data_collator, DataCollatorForSeq2Seq):
                    data_collator = DataCollatorForSeq2Seq(__tokenizer.tokenizer)
                else:
                    data_collator = TransformersDataCollatorForLanguageModeling(__tokenizer.tokenizer, mlm = False, mlm_probability = 0.0)
        other_metrics = []
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('sft_trainer', other_metrics)
        IGNORED_TOKENIZER_NAMES = os.environ.get('UNSLOTH_IGNORED_TOKENIZER_NAMES', '').split('\n')
        from unsloth_zoo.tokenizer_utils import fix_untrained_tokens
        from unsloth_zoo.training_utils  import fix_zero_training_loss
        if 'tokenizer' not in locals(): tokenizer = processing_class
        fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)
        fix_zero_training_loss(model, tokenizer, train_dataset)
        
        super().__init__(
            model = model,
            args = args,
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            model_init = model_init,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            peft_config = peft_config,
            dataset_text_field = dataset_text_field,
            packing = packing,
            formatting_func = formatting_func,
            max_seq_length = max_seq_length,
            infinite = infinite,
            num_of_sequences = num_of_sequences,
            chars_per_token = chars_per_token,
            dataset_num_proc = dataset_num_proc,
            dataset_batch_size = dataset_batch_size,
            neftune_noise_alpha = neftune_noise_alpha,
            model_init_kwargs = model_init_kwargs,
            dataset_kwargs = dataset_kwargs,
            eval_packing = eval_packing,**kwargs)
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        if hasattr(self, 'accelerator'):
            scaler = self.accelerator.scaler
            current_model = model
            while hasattr(current_model, 'model'):
                current_model.accelerator_scaler = scaler
                current_model = current_model.model
            current_model.accelerator_scaler = scaler
        pass
        
pass
