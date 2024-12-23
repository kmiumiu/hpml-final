# Parameter Efficient Fine-tuning of LLMs with Limited Resources

## Prerequisites

Install dependencies:

```shell
pip install -r requirements.txt
```

Configure accelerate:

```shell
accelerate config
```

To use Llama 3.2, you will need to request access first 
<https://huggingface.co/meta-llama/Llama-3.2-1B>

Log in to HuggingFace Hub:

```shell
huggingface-cli login
```

## Training

All scripts have been configured for use on Terremoto.

To train Llama-3.2-1B using the Stanford Alpaca configuration:

```shell
sbatch train_alpaca.sh
```
This will run the original training script from the Stanford Alpaca paper with the Llama-3.2-1B model.

All other scripts can be found in the `scripts` directory.

```
sbatch {script}.sh
```

`scripts/train.py` is the main training script, utilizing Accelerate with Deepspeed. 

## Ablation Test Configurations

| Configuration | Quantization | LoRA Rank | ZeRO Stage |
| --- | --- | --- | --- |
| Baseline | FP16 | None | None |
| LoRA Only | FP16 | 8 | None |
| QLoRA | 4-bit | 8 | None |
| ZeRO 1 | 4-bit | 8 | 1 |
| ZeRO 2 | 4-bit | 8 | 2 |
| ZeRO 3 | 4-bit | 8 | 3 |

## Results

All results are logged to wandb, please email me if you need access.

Due to issues with Terremoto, we couldn't get Accelerate/Deepspeed working.

Baseline training using the Stanford Alpaca configuration took 9 hours 25 minutes on Terremoto with 2 V100 GPUs.
Wandb logs shows near maximum GPU utilization throughout the run across both GPUs. 

[GPU Utilization](https://api.wandb.ai/links/km3635-columbia-university/r16yuwm5)

Oddly, the training loss seems to plateau each epoch, then sharply dropping at the boundaries. Given that the learning rate plot is smoothly declining, it seems unlikely that the optimizer is the issue. More likely is that there is some interal issue with logging where the training loss is not being logged accurately at the step-level. 

[Training Loss](https://wandb.ai/km3635-columbia-university/huggingface/reports/train-loss-24-12-22-19-11-53---VmlldzoxMDY5NzUwNg)

[Learning Rate](https://api.wandb.ai/links/km3635-columbia-university/mirs3m37)

