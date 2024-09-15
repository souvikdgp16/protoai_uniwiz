import argparse
import time
import csv
import tqdm
import os
import json

import sys
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# from transformers import StoppingCriteriaList, LLamaQaStoppingCriteria


class Common:
    def __init__(self, model_name, lora_weight_path, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name, lora_weight_path)

    def load_model(self, model_name, lora_weight_path):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        lora_model = PeftModel.from_pretrained(
            base,
            lora_weight_path,
            torch_dtype=torch.float16,
        )

        print("Applying LoRA")
        model = lora_model.merge_and_unload()

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        # self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        # self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, verbose=True,
                 remove_stop_words=False, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                          output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                                          top_p=top_p, top_k=top_k, temperature=temperature,
                                          stopping_criteria=self.stopping_criteria, **kwargs)
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, None

    def lm_score(self, input_text1, input_text2, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            outputs = self.model(input_ids)[0].squeeze(0)
            outputs = outputs.log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs, None
