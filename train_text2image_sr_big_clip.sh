#! /bin/bash

WORLD_SIZE=${SLURM_NTASKS:-1}
RANK=${SLURM_PROCID:-0}
# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$SLURM_NODELIST" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=27878
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
    --experiment-name pretrain_inf_dit \
    --image-block-size 128 \
    --mode pretrain --train-iters 1000000 \
    --eval-interval 1000 --eval-iters 1 \
    --save ./ckpt --save-interval 5000 \
    --log-interval 20 \
    --train-data ';WDS_DIR1,WDS_DIR2' \
    --valid-data ';WDS_DIR' \
    --split 1,0,0 \
    --num-layers 28 \
    --vocab-size 1 \
    --hidden-size 1280 \
    --nogate --no-crossmask \
    --input-time adaln \
    --num-attention-heads 16 \
    --hidden-dropout 0. \
    --attention-dropout 0. \
    --text-dropout 0.1 \
    --in-channels 6 \
    --out-channels 3 \
    --image-size 512 \
    --patch-size 4 \
    --config-path configs/text2image-sr.yaml \
    --max-sequence-length 256 \
    --layernorm-epsilon 1e-6 \
    --layernorm-order 'pre' \
    --model-parallel-size 1 \
    --tokenizer-type 'fake' \
    --iterable-dataset \
    --no-load-rng \
    --deepspeed \
    --random-position \
    --qk-ln \
    --deepspeed_config ${config_json} \
    --seed $RANDOM \
    --image-condition \
    --vector-dim 768 \
    --lr-dropout 0.07 \
    --re-position \
    --num-workers 6 \
    --cross-lr \
"

run_cmd="WORLD_SIZE=$WORLD_SIZE RANK=$RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT LOCAL_RANK=$LOCAL_RANK LOCAL_WORLD_SIZE=8 python train_text2image_sr.py ${options}"

echo ${run_cmd}
eval ${run_cmd}

set +x
echo "DONE on `hostname`"