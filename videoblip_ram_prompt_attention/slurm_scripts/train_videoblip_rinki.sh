#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 1 node to this job.
#SBATCH --ntasks=1 # Allocate *at most* 1 task for job steps in the job
#SBATCH --cpus-per-task=1 # Each task needs only one CPU
#SBATCH --mem=12G # This particular job won't need much memory
#SBATCH --time=07-01:00:00  # 7 day and 1 minute 
#SBATCH --job-name="batch job test"
#SBATCH -p cpu # You could pick other partitions for other jobs
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.

#added by Rinki
n_present_video=2
n_future_video=20
ego4d_annot_path_train="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json"
ego4d_annot_path_test="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_val.json"
#ego4d_clip_path="/data/AmitRoyChowdhury/ego4d_data/v2/clips/"
forecast_annot_tr_dir="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/anticipation_final_train_annots.json"
forecast_annot_te_dir="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/anticipation_final_test_annots.json"
no_frame=8
#ram_tags_path="/data/AmitRoyChowdhury/Anirudh/ram_openset_tags_ego4d/"
#saved_tr_anot_file=out_dir+"/"+"anticipation_"+"train" + "_annots.json"
#saved_te_anot_file=out_dir+"/"+"anticipation_"+"test" + "_annots.json"




VIDS_DIR="/data/AmitRoyChowdhury/ego4d_data/v2/clips/"
LTA_ANNOTS_DIR="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/"
RAM_TAGS_PATH=" /data/AmitRoyChowdhury/Anirudh/ram_openset_tags_ego4d/"
SAVE_DIR="/data/AmitRoyChowdhury/Rinki/videoblip_model/" #########################changed by rinki
NUM_TRAIN_EPOCHS=5
NUM_WORKERS=8
# MODEL_NAME_OR_PATH="Salesforce/blip2-opt-2.7b"
PROCESSOR_PATH="Salesforce/blip2-opt-2.7b"
MODEL_NAME_OR_PATH="Salesforce/blip2-opt-2.7b" #"videoblip_output/checkpoint-999/"
OUTPUT_DIR="/data/AmitRoyChowdhury/Rinki/videoblip_ram_output/" ############# changed by rinki
TRAIN_BATCH_SIZE=16

export MASTER_NODE=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
#echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_NODE=$MASTER_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

RDZV_ID=$RANDOM3
MASTER_NODE=$(srun --nodes=1 --ntasks=1 hostname)
MASTER_NODE=$(hostname)

eval "$(conda shell.bash hook)"
conda activate llava ###############################################################################????????????

# torchrun --nnodes=1 --nproc_per_node=4 --rdzv-id=$RDZV_ID --rdzv-backend=c10d \
#     --rdzv-endpoint=$MASTER_NODE \

# torchrun --nproc_per_node=4 main.py  \
python3 main_rinki.py --n_present_video ${n_present_video} --n_future_video ${n_future_video} --no_frame ${no_frame} --forecast_annot_tr_dir ${forecast_annot_tr_dir} --forecast_annot_te_dir ${forecast_annot_te_dir} --vids_dir ${VIDS_DIR} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} --processor_path ${PROCESSOR_PATH} --ram_tags_path ${RAM_TAGS_PATH} \
    --lta_annots_dir ${LTA_ANNOTS_DIR} --output_dir ${OUTPUT_DIR} --save_dir ${SAVE_DIR} --num_train_epochs ${NUM_TRAIN_EPOCHS} --warmup_steps 1000 \
    --learning_rate 1e-5 --per_device_train_batch_size 16 --gradient_accumulation_steps 4 --ddp_find_unused_parameters False --per_device_eval_batch_size 16 --weight_decay 0.05 --dataloader_num_workers ${NUM_WORKERS} --bf16 True --evaluation_strategy epoch --eval_steps 200 --save_strategy epoch --save_steps 200 --save_total_limit 5 --logging_steps 10    
