#! /bin/bash

WORLD_SIZE=${SLURM_NTASKS:-1}
RANK=${SLURM_PROCID:-0}
# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$SLURM_NODELIST" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=$((RANDOM % 10001 + 20000))
else
    MASTER_ADDR=`scontrol show hostnames $SLURM_NODELIST | head -n 1`
    MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
fi
# generate a port at random
LOCAL_RANK=${SLURM_LOCALID:-0}
MP_SIZE=1

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

config_json="scripts/ds_config_zero_clip.json"

options=" \
    --input-type cli \
    --inference_type full \
    --block_batch 4 \
    --experiment-name generate \
    --mode inference \
    --inference-batch-size 1 \
    --image-size 512 \
    --input-time adaln \
    --nogate --no-crossmask \
    --bf16 \
    --num-layers 28 \
    --vocab-size 1 \
    --hidden-size 1280 \
    --num-attention-heads 16 \
    --hidden-dropout 0. \
    --attention-dropout 0. \
    --in-channels 6 \
    --out-channels 3 \
    --cross-attn-hidden-size 640 \
    --patch-size 4 \
    --config-path configs/text2image-sr.yaml \
    --max-sequence-length 256 \
    --layernorm-epsilon 1e-6 \
    --layernorm-order 'pre' \
    --model-parallel-size 1 \
    --tokenizer-type 'fake' \
    --random-position \
    --qk-ln \
    --out-dir samples \
    --network ckpt/mp_rank_00_model_states.pt \
    --round 32 \
    --init_noise \
    --image-condition \
    --vector-dim 768 \
    --re-position \
    --cross-lr \
    --seed $RANDOM \
    --infer_sr_scale 4 \
"

run_cmd="WORLD_SIZE=$WORLD_SIZE RANK=$RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT LOCAL_RANK=$LOCAL_RANK python generate_t2i_sr.py ${options}"

echo ${run_cmd}
eval ${run_cmd}

set +x
echo "DONE on `hostname`"