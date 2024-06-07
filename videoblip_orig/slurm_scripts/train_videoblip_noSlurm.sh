#!/bin/bash -l

# SLURM directives are removed as they are not needed

# Set variables
n_present_video=2
n_future_video=20
ego4d_annot_path_train="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json"
ego4d_annot_path_test="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_val.json"
forecast_annot_tr_dir="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/anticipation_final_train_annots.json"
forecast_annot_te_dir="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/anticipation_final_test_annots.json"
no_frame=8

VIDS_DIR="/data/AmitRoyChowdhury/ego4d_data/v2/clips/"
LTA_ANNOTS_DIR="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/"
RAM_TAGS_PATH="/data/AmitRoyChowdhury/Anirudh/ram_openset_tags_ego4d/"
SAVE_DIR="/data/AmitRoyChowdhury/Rinki/videoblip_orig_model/" #########################changed by rinki
NUM_TRAIN_EPOCHS=5
NUM_WORKERS=8
PROCESSOR_PATH="Salesforce/blip2-opt-2.7b"
MODEL_NAME_OR_PATH="Salesforce/blip2-opt-2.7b" #"videoblip_output/checkpoint-999/"
OUTPUT_DIR="/data/AmitRoyChowdhury/Rinki/videoblip_orig_output/" ############# changed by rinki
TRAIN_BATCH_SIZE=16

# Initialize environment variables for non-SLURM execution
MASTER_NODE=$(hostname)
WORLD_SIZE=1
echo "MASTER_NODE=$MASTER_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"

# Conda environment activation
eval "$(conda shell.bash hook)"
conda activate llava

# Run the main script using Python
python3 main.py \
    --n_present_video ${n_present_video} \
    --n_future_video ${n_future_video} \
    --no_frame ${no_frame} \
    --forecast_annot_tr_dir ${forecast_annot_tr_dir} \
    --forecast_annot_te_dir ${forecast_annot_te_dir} \
    --vids_dir ${VIDS_DIR} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --lta_annots_dir ${LTA_ANNOTS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --save_dir ${SAVE_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --warmup_steps 1000 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --ddp_find_unused_parameters False \
    --per_device_eval_batch_size 16 \
    --weight_decay 0.05 \
    --dataloader_num_workers ${NUM_WORKERS} \
    --bf16 True \
    --evaluation_strategy epoch \
    --eval_steps 200 \
    --save_strategy epoch \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10
