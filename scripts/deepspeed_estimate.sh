#!/bin/sh
#
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=Deepspeed    # The job name.
#SBATCH --gres=gpu:2
#SBATCH -c 12                    # The number of cpu cores to use.
#SBATCH --time=5:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.

python -c 'from transformers import AutoModelForCausalLM; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'

# End of script
