formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time


deepspeed --include localhost:0 finetune.py \
    --model_name_or_path output/WebQA_LoRA/20240711084950/checkpoint-20000 \
    --output_dir output/InterfereQA_LoRA/$formatted_time/ \
    --train_data_path processed_data/Cosmos_train.jsonl \
    --eval_data_path processed_data/Cosmos_valid.jsonl \
    --learning_rate 5e-5 --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 6  --model_max_length 1024 --bf16 --use_lora \
    --gradient_accumulation_steps 1 --warmup_steps 100 \
    --num_train_epochs 1 --weight_decay 0.01 \
    --evaluation_strategy steps --eval_steps 500 \
    --save_strategy steps --save_steps 500 --seed 42 \
    --log_level info --logging_strategy steps --logging_steps 10 \
    --deepspeed configs/ds_config_zero3_offload.json
