import os
import pandas as pd
import torch
import pickle
import argparse
import random
import sys
import torch.nn as nn
# import bitsandbytes as bnb
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback, IntervalStrategy


# from accelerate import FullyShardedDataParallelPlugin, Accelerator
# from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
# fsdp_plugin = FullyShardedDataParallelPlugin(
#     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
#     optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
# )
# accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

def load_data(path):
    with open(
            path,
            'rb') as handle:
        data = pickle.load(handle)
    return data


def load_model(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        device_map=args.device_map,
        quantization_config=bnb_config
    )
    total_params, params = 0, 0

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path, add_eos_token=True
    )

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    config.save_pretrained(args.output_path)

    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0

    for n, p in model.model.named_parameters():
        if any([x in n for x in ["lora"]]):
            total_params += p.numel()
        params += p.numel()

    print(
        "Total number of parameters: {}M, rate: {}%".format(
            total_params // 1000 / 1000, round(total_params / params * 100, 2)
        )
    )

    return model, tokenizer


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        result = self.tokenizer(
            row["input"],
            truncation=True,
            max_length=self.args.cutoff_len + 1,
            padding=True,
        )

        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            # "labels": result["input_ids"].copy()
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--pretrained_model_path", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--val_set_ratio", type=float, default=0.05)
    parser.add_argument("--target_modules", type=list, default=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ])
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="1")

    args, extra = parser.parse_known_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    GRADIENT_ACCUMULATION_STEPS = args.batch_size // args.micro_batch_size

    if ddp:
        args.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    model, tokenizer = load_model(args)
    data = load_data(args.dataset_path)
    data = pd.DataFrame(data)  # .drop_duplicates()

    train_df, test_df = train_test_split(data, test_size=args.val_set_ratio)

    train_data = SFTDataset(train_df.sample(frac=1), tokenizer, args)
    val_data = SFTDataset(test_df.sample(frac=1), tokenizer, args)

    # Training
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            optim="paged_adamw_8bit",
            bf16=True,
            logging_steps=20,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            output_dir=args.output_path,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            save_total_limit=args.save_total_limit,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    # ).__get__(model, type(model))
    #
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer.train()

    model.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
