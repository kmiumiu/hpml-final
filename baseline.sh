#!/bin/sh
#
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=Ablation    # The job name.
#SBATCH --gres=gpu:2
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=30:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.
 
torchrun --nproc_per_node=1 --master_port=8080 stanford_alpaca/train.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
    --data_path stanford_alpaca/alpaca_data.json \
    --fp16 True \
    --output_dir "./results" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
# End of script
