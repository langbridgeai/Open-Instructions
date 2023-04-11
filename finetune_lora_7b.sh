#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

run_name=ailurus-7b
output_dir=${project_dir}/${run_name}/output_models/
log_dir=${project_dir}/${run_name}/logs/

model_name_or_path=${}
dataset_path=${}

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  fastchat/train/train_lora.py \
    --deepspeed ailurus/configs/ds_config_zero2.json \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${dataset_path} \
    --lazy_preprocess True \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --run_name ${run_name} \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --evaluation_strategy "no" \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 20 \
    --tf32 True \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
