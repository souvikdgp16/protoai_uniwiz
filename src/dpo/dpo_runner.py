# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: you need to install transformers from main to run this script. See https://huggingface.co/docs/transformers/installation#install-from-source
# TODO: bump transformers version in requirements at next release.

# 0. imports
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import pandas as pd

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig

from trl import DPOTrainer
from peft import PeftModel
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    device: Optional[str] = field(default="cuda", metadata={"help": "the model name"})
    num_gpus: Optional[str] = field(default="auto", metadata={"help": "the model name"})
    max_gpu_memory: Optional[int] = field(default=20, metadata={"help": "max length of each sample"})
    lora_weight_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})

    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=1500, metadata={"help": "max number of training steps"})
    # lora parameters
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the r parameter of the LoRA adapters"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "optimizer learning rate"})
    #target_modules = ['q_proj', "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","lm_head"]
    target_modules = ["gate_proj", "down_proj", "up_proj"]
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
                    '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
                    'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
                    "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )

    output_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "max length of each sample's prompt"})


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt):],
            "rejected": sample["rejected"][len(prompt):],
        }

    col_list = ["prompt", "chosen", "rejected"]

    if split == "train":
        a_hh = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
        df_hh = a_hh.to_pandas().sample(3000, random_state=42)
        ds_hh = Dataset.from_pandas(df_hh)
        # if sanity_check:
        #     dataset = dataset.select(range(min(len(dataset), 1000)))

        ds_hh = ds_hh.map(split_prompt_and_responses)
        #
        # intel_dpo = load_dataset("Intel/orca_dpo_pairs")
        # df_intel_dpo = intel_dpo["train"].to_pandas()[:20000].sample(6000, random_state=42)
        # df_intel_dpo["prompt"] = df_intel_dpo.apply(lambda x: x["system"] + "\n" + x["question"], axis=1)
        #
        # df = pd.concat([ds_hh.to_pandas()[col_list], df_intel_dpo[col_list]])
        #
        # return Dataset.from_pandas(df)
        return ds_hh
    else:
        a_hh = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
        df_hh = a_hh.to_pandas().sample(1000, random_state=42)
        ds_hh = Dataset.from_pandas(df_hh)
        # if sanity_check:
        #     dataset = dataset.select(range(min(len(dataset), 1000)))

        ds_hh = ds_hh.map(split_prompt_and_responses)

        # intel_dpo = load_dataset("Intel/orca_dpo_pairs")
        # df_intel_dpo = intel_dpo["train"].to_pandas()[-20000:].sample(500, random_state=42)
        # df_intel_dpo["prompt"] = df_intel_dpo.apply(lambda x: x["system"] + "\n" + x["question"], axis=1)
        #
        # df = pd.concat([ds_hh.to_pandas()[col_list], df_intel_dpo[col_list]])
        #
        # return Dataset.from_pandas(df)
        return ds_hh


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1  # world_size = 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  # auto
    #     gradient_accumulation_steps = script_args.gradient_accumulation_steps // world_size
    #     print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # 1. load a pretrained model
    if script_args.device == "cuda":
        kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{script_args.model_name_or_path}/offload"}
        if script_args.num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            script_args.num_gpus = int(script_args.num_gpus)
            if script_args.num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: f"{script_args.max_gpu_memory}GiB" for i in range(script_args.num_gpus)},
                })
    elif script_args.device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {script_args.device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    base = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path,
                                                quantization_config=bnb_config,
                                                **kwargs)

    lora_model = PeftModel.from_pretrained(
        base,
        script_args.lora_weight_path,
        torch_dtype=torch.float16,
    )

    print("Applying LoRA")
    model = lora_model.merge_and_unload()

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules=script_args.target_modules,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    config.save_pretrained(script_args.output_path)

    model = get_peft_model(model, config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Anthropic Helpful-Harmless dataset
    train_dataset = get_hh("train", sanity_check=script_args.sanity_check)

    # 3. Load evaluation dataset
    eval_dataset = get_hh("test", sanity_check=script_args.sanity_check)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        max_steps=script_args.max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        output_dir=script_args.output_path,
        # save_total_limit=script_args.save_total_limit,
        load_best_model_at_end=True,
        optim="paged_adamw_8bit",
        warmup_steps=150,
        report_to=script_args.report_to,
        bf16=True,
        gradient_checkpointing=script_args.gradient_checkpointing,
        # TODO: uncomment that on the next transformers release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        # loss_type="kto_pair",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=True,
        peft_config=config,
    )

    # 6. train
    dpo_trainer.train()
