export NCCL_IB_CUDA_SUPPORT=1
# export PYTHONPATH=$PYTHONPATH:$(pwd)/lm_engine
export PYTHONPATH=$(pwd)/lm_engine/lm_engine/data/megatron/utils/build:$PYTHONPATH

DEFAULT_TRITON_PTXAS_PATH="/data/cl/user/zfw/miniconda3/envs/width-varying-lm-b300/bin/ptxas"
if [ -z "${TRITON_PTXAS_PATH:-}" ] && [ -x "$DEFAULT_TRITON_PTXAS_PATH" ]; then
    export TRITON_PTXAS_PATH="$DEFAULT_TRITON_PTXAS_PATH"
    echo "Using TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"
fi

# ---------------------------------------------------
# 1. Smart Network Configuration
# ---------------------------------------------------
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)

if [ "$NNODES" -eq 1 ]; then
    # SINGLE NODE: Always use loopback.
    # It is faster, safer, and bypasses firewalls.
    echo "Detected single-node run. Using loopback."
    export NCCL_SOCKET_IFNAME="lo"
    export GLOO_SOCKET_IFNAME="lo"
    MASTER_ADDRESS="127.0.0.1"
else
    # MULTI-NODE: Use the high-speed interface.
    echo "Detected multi-node run. Using bond0."
    export NCCL_SOCKET_IFNAME="bond0"
    export GLOO_SOCKET_IFNAME="bond0"
    MASTER_ADDRESS="9.47.193.81" # In a real cluster, this usually comes from $SLURM_JOB_NODELIST or similar
fi

# ---------------------------------------------------
# 2. Port Selection
# ---------------------------------------------------
# Random ports (20000-30000) are usually blocked by cluster firewalls on bond0.
# If running multi-node, use a specific port allowed by your admin, or rely on the scheduler.
# For single node (lo), random ports are fine.
min=20000
max=30000
MASTER_PORT=$(( RANDOM % (max + 1 - min) + min ))

# ---------------------------------------------------
# 3. Execution
# ---------------------------------------------------
if [ -z "${1}" ]; then
    echo "Error: config file required"
    echo "Usage: bash pretrain.sh <config.yml>"
    exit 1
fi
if [ ! -f "${1}" ]; then
    echo "Error: config file not found: ${1}"
    exit 1
fi

TOKENIZERS_PARALLELISM=false \
NCCL_DEBUG=WARN \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=101 \
    --rdzv_endpoint=$MASTER_ADDRESS:$MASTER_PORT \
    --rdzv_backend=c10d \
    -m pretrain \
    --config "${1}"
