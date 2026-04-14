#!/usr/bin/env bash
# LEASH: Qwen3-4B-Thinking, target length 12k

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export RAY_PORT=6379

# ============================================================
# USER CONFIG: Set these paths before running
# ============================================================
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
WORK_DIR=${WORK_DIR:-"${REPO_DIR}"}
cd "$WORK_DIR"

project_name='LEASH'
exp_name='leash-qwen3-4b-target12k'

# ============================================================
# Ray cluster setup
# ============================================================
if [[ "$NODE_RANK" -eq 0 ]]; then
    ray stop
    ray start --head --port=$RAY_PORT --dashboard-port=8265 --dashboard-host=0.0.0.0
    sleep 10

    while true; do
        ray_status_output=$(ray status)
        active_nodes=$(echo "$ray_status_output" | awk '/Active:/ {flag=1; next} /Pending:/ {flag=0} flag' | grep -oE '[a-f0-9]{40}' | wc -l)
        echo "Connected nodes: ${active_nodes:-0}/${NNODES}, waiting..."
        if [[ "$active_nodes" -eq "$NNODES" ]]; then
            echo "All nodes connected."
            break
        fi
        sleep 5
    done
else
    ray stop
    sleep 10
    while true; do
        ray start --address=$MASTER_ADDR:$RAY_PORT
        if ray status &>/dev/null; then
            echo "Joined cluster."
            while true; do
                if ! ray status --address=$MASTER_ADDR:$RAY_PORT &>/dev/null; then
                    echo "Head node stopped, exiting."
                    ray stop
                    exit 0
                fi
                sleep 10
            done
        else
            echo "Failed to join cluster, retrying in 10s..."
            ray stop
            sleep 10
        fi
    done
fi

# ============================================================
# Algorithm settings
# ============================================================
adv_estimator=grpo

# KL settings (disabled, same as DAPO)
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# PPO clipping
clip_ratio_low=0.2
clip_ratio_high=0.28

# Sequence lengths
max_prompt_length=1024
max_response_length=32768

# Overlong buffer (disabled)
enable_overlong_buffer=False

loss_agg_mode="token-mean"

# Batch sizes
train_prompt_bsz=64
gen_prompt_bsz=64
val_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=32
filter_groups_enable=False
filter_groups_metric=acc
max_num_gen_batches=5

# ============================================================
# Paths: override via environment variables
# ============================================================
MODEL_PATH=${MODEL_PATH:?"Please set MODEL_PATH to your Qwen3-4B-Thinking checkpoint"}
CKPTS_DIR=${CKPTS_DIR:-"${REPO_DIR}/train_results/${project_name}/${exp_name}"}
TRAIN_FILES=${TRAIN_FILES:-"['${REPO_DIR}/data/train/4k_high_quality_deepmath.parquet']"}

# Validation data paths (AIME24, AIME25, HMMT25, AMC, GPQA)
EVAL_DIR="${REPO_DIR}/data/eval"
VAL_FILES=${VAL_FILES:-"['${EVAL_DIR}/valid.aime24.parquet','${EVAL_DIR}/valid.aime25.parquet','${EVAL_DIR}/valid.hmmt25.parquet','${EVAL_DIR}/valid.amc.parquet','${EVAL_DIR}/valid.gpqa.parquet']"}

# Sampling settings
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0.6
val_top_p=0.95
val_top_k=-1

# Performance
sp_size=1
actor_ppo_max_token_len=34816
infer_ppo_max_token_len=48000
offload=True
fsdp_size=-1

# ============================================================
# LEASH settings (paper best: 4B Lt=12k)
# ============================================================
use_leash=True
target_length=12288
lambda_init=0.2
lambda_lr=0.005
lambda_max=1.00
leash_type="average"

# ============================================================
# Launch training
# ============================================================
if [[ "$NODE_RANK" -eq 0 ]]; then
    python3 -m recipe.leash.main_dapo \
        data.train_files="${TRAIN_FILES}" \
        data.val_files="${VAL_FILES}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.gen_batch_size=${gen_prompt_bsz} \
        data.train_batch_size=${train_prompt_bsz} \
        data.val_batch_size=${val_prompt_bsz} \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        algorithm.use_leash=${use_leash} \
        algorithm.leash_config.target_length=${target_length} \
        algorithm.leash_config.lambda_init=${lambda_init} \
        algorithm.leash_config.lambda_lr=${lambda_lr} \
        algorithm.leash_config.lambda_max=${lambda_max} \
        algorithm.leash_config.leash_type=${leash_type} \
        algorithm.filter_groups.enable=${filter_groups_enable} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        custom_reward_function.path="./utils/reward_utils/my_reward_func.py" \
        custom_reward_function.name="my_reward_func_with_timeout" \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=8192 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=8 \
        trainer.val_before_train=False \
        trainer.test_freq=5 \
        trainer.save_freq=5 \
        trainer.total_epochs=5 \
        trainer.total_training_steps=100 \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=auto \
        trainer.log_val_generations=10
    ray stop
fi
