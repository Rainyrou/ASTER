#!/usr/bin/env bash
set -euo pipefail
nodes=(
    ""
)
MASTER_ADDR="${nodes[0]}"
MASTER_PORT=29500
USER=${USER}
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
REMOTE_DIR="$CODE_DIR"

# 枚举多个yaml配置文件，分别使用不同的数据集进行训练
YAML_CONFIGS=(
    "$CODE_DIR/examples/train_full/gpt-oss.yaml"
    "$CODE_DIR/examples/train_full/gpt-oss_min.yaml"
    # "$CODE_DIR/examples/train_full/Open-AgentRL.yaml"
)

# >>> 1. 根据你的 conda 安装位置修改 <<<
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
CONDA_BASE="/root/miniconda3"
CONDA_INIT="$CONDA_BASE/etc/profile.d/conda.sh"
__conda_setup="$("$CONDA_BASE/bin/conda" 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_INIT" ]; then
        . "$CONDA_INIT"
    else
        export PATH="$CONDA_BASE/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# ------------------------------------------------------------------------------

# 遍历每个yaml配置文件进行训练
for YAML in "${YAML_CONFIGS[@]}"; do
    echo "========================================================================"
    echo "开始训练配置: $YAML"
    echo "========================================================================"
    
    echo ">>>> 同步代码到所有节点 ..."
    for n in "${nodes[@]}"; do
        [[ $n == "$MASTER_ADDR" ]] && continue
        rsync -avz --exclude='.git' --exclude='ckpts' --exclude='*.log' \
              -e ssh "$CODE_DIR/" "${USER}@${n}:${REMOTE_DIR}/"
    done

    PIDS=()
    LOG_DIR="$CODE_DIR/ckpts"
    mkdir -p "$LOG_DIR"
    
    # 从yaml文件名提取数据集名称用于日志
    YAML_BASENAME=$(basename "$YAML" .yaml)

    for rank in "${!nodes[@]}"; do
        ip=${nodes[rank]}
        timestamp=$(date +%Y%m%d_%H%M%S)
        log_name="train_sft_${YAML_BASENAME}_${timestamp}_rank${rank}.log"
        
        if [[ $ip == "$MASTER_ADDR" ]]; then
            log="$LOG_DIR/$log_name"
        else
            log="$REMOTE_DIR/ckpts/$log_name"
        fi

        # >>> 2. 关键：conda 激活后再训练 <<<
        cmd="source \"$CONDA_INIT\" && \
             conda activate llamafactory && \
             nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9 2>/dev/null || true && \
             cd \"$REMOTE_DIR\" && \
             mkdir -p \"$REMOTE_DIR/ckpts\" && \
             FORCE_TORCHRUN=1 \
             NNODES=${#nodes[@]} \
             NODE_RANK=$rank \
             MASTER_ADDR=$MASTER_ADDR \
             MASTER_PORT=$MASTER_PORT \
                 llamafactory-cli train \"$YAML\" > \"$log\" 2>&1"

        echo ">>>> 在 $ip 启动 rank=$rank 训练，日志：$log"
        if [[ $ip == "$MASTER_ADDR" ]]; then
            # 本机后台
            bash -c "$cmd" &
        else
            # 远程后台
            ssh "${USER}@${ip}" "nohup bash -c '$cmd' </dev/null >/dev/null 2>&1 &"
        fi
        PIDS+=($!)
    done

    echo ">>>> 所有节点训练已投递，等待完成 ..."
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    echo ">>>> 配置 $YAML_BASENAME 训练完成！"
    echo ""
done

echo "========================================================================"
echo "所有配置训练结束！"
echo "========================================================================"


