#!/usr/bin/env bash
set -Eeuo pipefail
set -x

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export VERL_AUTO_PADDING=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NCCL_DEBUG=INFO

###### SET YOUR WANDB CONFIG HERE ######
export WANDB_API_KEY=TODO
#######################################

ulimit -n 65535
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/aster/config"

##### System Configuration ######
n_nodes=1
n_gpu=8
num_workers=8

##### Model Configuration ######
model_path=/root/model_to_eval
model_name=Aster_4B_Eval

##### Dataset Configuration ######
train_datasets=$PROJECT_DIR/datasets/dapo_filter_all_correct.parquet
val_datasets="[$PROJECT_DIR/datasets/aime25.parquet,$PROJECT_DIR/datasets/hmmt25.parquet]"

##### Sequence Length Configuration ######
max_prompt_length=2048
max_response_length=92160
max_num_batched_tokens=$((max_prompt_length+max_response_length))
max_assistant_turns=256

##### Training Hyperparameters ######
train_bsz=128
ppo_mini_batch_size=64
ppo_max_token_len_per_gpu=$((max_prompt_length*8+max_response_length*8))
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"

##### Parallelism Configuration ######
sp_size=1
gen_tp=1
rollout_n=8

##### Rollout Engine Configuration ######
gpu_memory_utilization=0.90
free_cache_engine=True
enforce_eager=False

##### Path and Experiment Configuration ######
save_path="$PROJECT_DIR/final_ckpts/${model_name}_maxturns_${max_assistant_turns}_maxresp_${max_response_length}"
experiment_name=${model_name}_maxturns_${max_assistant_turns}_maxresp_${max_response_length}
export TENSORBOARD_DIR="${save_path}/tensorboard"
mkdir -p "$save_path" "$save_path/rollout_data" "$save_path/tensorboard"

python3 -m aster.main \
    --config-path="$CONFIG_PATH" \
    --config-name='aster' \
    +ray_kwargs.ray_init.address=auto \
    +ray_kwargs.ray_init.runtime_env.working_dir="$PROJECT_DIR" \
    +ray_kwargs.ray_init.runtime_env.excludes='[".git","final_ckpts","datasets","wandb",".venv","__pycache__","*.zip","llamafactory/*","sandboxfusion/*"]' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$train_bsz \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.custom_cls.path=$PROJECT_DIR/aster/custom_dataset.py \
    data.custom_cls.name=CustomRLHFDataset \
    data.tool_config_path="$PROJECT_DIR/aster/config/tool_config/sandbox_fusion_tool_config.yaml" \
    data.enable_thinking=True \
    custom_reward_function.path=aster/reward_manager/reward_score.py \
    custom_reward_function.name=compute_score \
    data.train_files=$train_datasets \
    data.val_files=$val_datasets \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.tis_imp_ratio_cap=2.0 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.agent.num_workers=$num_workers \
    actor_rollout_ref.rollout.enable_thinking=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enforce_eager=$enforce_eager \
    actor_rollout_ref.rollout.free_cache_engine=$free_cache_engine \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/aster/config/tool_config/sandbox_fusion_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_assistant_turns \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name='aster_rl' \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=True \
    trainer.eval_only=True \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$save_path \
    trainer.rollout_data_dir=$save_path/rollout_data \
    trainer.validation_data_dir=$save_path/val_rollout_data \
    trainer.total_epochs=6 "$@" \
    2>&1 | tee "$save_path/eval.log"

