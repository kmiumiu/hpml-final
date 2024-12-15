#!/bin/sh
#
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=Deepspeed    # The job name.
#SBATCH --gres=gpu:2
#SBATCH -c 12                    # The number of cpu cores to use.
#SBATCH --time=9:30:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.

python -c 'from transformers import AutoModelForCausalLM; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'

python -c 'from transformers import AutoModelForCausalLM; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'

torchrun --nproc_per_node=2 --master_port=1234 train.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
    --data_path ./alpaca_data.json \
    --fp16 True \
    --output_dir "./deepspeed" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
# End of script
