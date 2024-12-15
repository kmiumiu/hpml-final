#!/bin/sh
#
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=Baseline    # The job name.
#SBATCH --gres=gpu:2
#SBATCH -c 12                    # The number of cpu cores to use.
#SBATCH --time=30:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.
 
torchrun --nproc_per_node=2 --master_port=1234 train.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
    --data_path ./alpaca_data.json \
    --fp16 True \
    --output_dir "./baseline" \
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
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' 
# End of script
