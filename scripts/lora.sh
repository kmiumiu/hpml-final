#!/bin/sh
#
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=Deepspeed    # The job name.
#SBATCH --gres=gpu:2
#SBATCH -c 12                    # The number of cpu cores to use.
#SBATCH --time=12:00:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.

accelerate launch --config_file=../accelerate_config/fsdp.yaml --num_processes 2 train.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
    --fp16 True \
    --output_dir "./lora" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --use_peft True \
    --logging_steps 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules "all-linear" \
    --report_to "wandb"

# End of script
