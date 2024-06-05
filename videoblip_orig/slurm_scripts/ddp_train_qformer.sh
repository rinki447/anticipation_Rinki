#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 1 node to this job.
#SBATCH --ntasks=1 # Allocate *at most* 1 task for job steps in the job
#SBATCH --cpus-per-task=1 # Each task needs only one CPU
#SBATCH --mem=5G # This particular job won't need much memory
#SBATCH --time=1-00:01:00  # 1 day and 1 minute 
#SBATCH --job-name="batch job test"
#SBATCH -p cpu # You could pick other partitions for other jobs
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.

BLIP_CAPTIONS_PATH="/home/csres/athatipelli/blip2_captions/"
VIDS_DIR="/data/AmitRoyChowdhury/ego4d_data/v2/clips/"
LTA_ANNOTS_DIR="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/"
SAVE_DIR="blip2_caps_model_ddp/"
TRAIN_GT_CAPTION_PATH="/home/csres/athatipelli/blipformer_captions_idea/train_clip_seg_narr.pkl"
VAL_GT_CAPTION_PATH="/home/csres/athatipelli/blipformer_captions_idea/val_clip_seg_narr.pkl"
NUM_TRAIN_EPOCHS=5
NUM_WORKERS=16
MODEL_NAME_OR_PATH="Salesforce/blip2-opt-2.7b"
OUTPUT_DIR="blipformer_output_ddp/"
# BLIP2_VISION_FEATS_DIR="/data/AmitRoyChowdhury/blip2_vision_feats/"

export MASTER_NODE=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

RDZV_ID=$RANDOM3
MASTER_NODE=$(srun --nodes=1 --ntasks=1 hostname)
MASTER_NODE=$(hostname)

export PYTHONPATH="${PYTHONPATH}:/home/csres/athatipelli/EILEV/"
conda activate eilev

# torchrun --nnodes=1 --nproc_per_node=4 --rdzv-id=$RDZV_ID --rdzv-backend=c10d \
#     --rdzv-endpoint=$MASTER_NODE \
# python3 main.py \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --rdzv-id=$RDZV_ID \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$MASTER_NODE \
python3 main.py \
    --vids_dir ${VIDS_DIR} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --lta_annots_dir ${LTA_ANNOTS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --save_dir ${SAVE_DIR} \
    --blip_captions_dir ${BLIP_CAPTIONS_PATH} \
    --train_gt_caption_path ${TRAIN_GT_CAPTION_PATH} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --val_gt_caption_path ${VAL_GT_CAPTION_PATH} \
    --warmup_steps 0 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --ddp_find_unused_parameters False \
    --per_device_eval_batch_size 32 \
    --weight_decay 0.05 \
    --dataloader_num_workers ${NUM_WORKERS} \
    --bf16 True \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10    


# Test

# python3 test_model.py --val_lavila_narrations_dir ${LAVILA_NARRATIONS_PATH}"/val_samples/" \
#     --prompts_dir ${PROMPTS_PATH} --ego4d_lta_annots_path ${LTA_ANNOTS_DIR}"/fho_lta_val.json" \
#     --checkpoint_path ${CHECKPOINT_PATH}
