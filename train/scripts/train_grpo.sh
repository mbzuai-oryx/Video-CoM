#!/bin/bash

MODEL_NAME="MBZUAI/Video-CoM-SFT"
DATA_PATH="/path/to/train_1p5k_v1_ytb_and_1p5k_generic_grpo_data.json"
DATA_FOLDER="/path/to/data_folder/containing/corresponding/media/files"
OUTPUT_DIR="/path/to/output_directory/to/save/checkpoints"

PER_DEVICE=1
GRAD_ACCUM=1

torchrun --nproc-per-node=8 src/train/train_grpo.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $DATA_FOLDER \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --num_generations 8 \
    --beta 0.04 \
    --per_device_train_batch_size $PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_completion_length 512 \
    --max_prompt_length 3276800 \
    --video_max_pixels $((360 * 420)) \
    --image_max_pixels $((128 * 28 * 28)) \
    --fps 1.0 \
    --learning_rate 1e-6 \
    --remove_unused_columns False \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --reward_llm_judge "None" --reward_func_names "reasoning_reward" --reward_weights "1.0"
