#    References:
#    HuggingFace TRL Documentation
#    https://huggingface.co/docs/trl/v0.13.0/en/sft_trainer
#    
#    Unsloth Llama3.1(8B) Finetuning Tutorial
#    https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=LjY75GoYUCB8

import torch
import transformers
import utils
from torch.utils.data import Dataset

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from datasets import load_dataset


max_seq_length = 512

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


def train():
    parser = TrlParser((SFTConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    print(training_args, model_args)


    torch_dtype = (
            model_args.torch_dtype 
            if model_args.torchdtype in [None, "auto"] 
            else getattr(torch, model_args.torchdtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        output_dir=training_args.output_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=max_seq_length,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
        pad_token=tokenizer.eos_token
    )
    peft_config = get_peft_config(model_args)

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
            model=model, 
            tokenizer=tokenizer, 
            train_dataset=dataset,
            dataset_text_field="text",
            peft_config=peft_config
            max_seq_length=max_seq_length,
            packing=False,
            args=training_args
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
