#!/bin/sh

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / ($BATCH_PER_DEVICE * $NUM_DEVICES * $SLURM_NNODES)))

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

DATA_PATH1="/path/to/activitynet_temporal_loc_reasoning_frame_chat.json"
DATA_PATH2="/path/to/train_v4_sft_mcq_15k_compact_v2_with_reasoning_prompt_and_ytb_1p4k.json"
DATA_PATH3="/path/to/viscot_spatail_train_128k.json"

DATA_FOLDER="/path/to/data_folder/containing/corresponding/media/files"

OUTPUT_DIR="/path/to/output_directory/to/save/checkpoints"

torchrun --nproc-per-node=4 src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH1|$DATA_PATH2|$DATA_PATH3|$DATA_PATH2" \
    --multi_batch_size "4,4,16,4" \
    --image_folder $DATA_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((360 * 420)) \
    --image_max_pixels $((128 * 28 * 28)) \
    --fps 1.0 \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --dataloader_num_workers 4
