import pandas as pd
import torch
from ast import literal_eval
from tqdm import tqdm, tqdm_notebook
import os
import anthropic
import re
import time
from pprint import pprint
from datasets import load_dataset
import argparse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForCausalLM, AutoTokenizer

tqdm.pandas()
sid_obj = SentimentIntensityAnalyzer()

# import spacy
# from sentence_transformers import SentenceTransformer, util

# sbert = SentenceTransformer('nli-distilroberta-base-v2')
# nlp = spacy.load('en_core_web_sm')

ant_client = anthropic.Client(
    api_key="")


def anthropic_api_response(prompt):
    model = "claude-instant-1.2"
    try:
        response = ant_client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model,
            max_tokens_to_sample=1000,
        )
        # print(response.completion)
        return response.completion
    except Exception as e:
        print(e)
        print("Will sleep for 45 sec")
        time.sleep(45)

        try:
            response = ant_client.completions.create(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model=model,
                max_tokens_to_sample=1000,
            )
            return response.completion
        except Exception as e:
            print(e)
            print("Will sleep for 90 sec")
            time.sleep(90)

            response = ant_client.completions.create(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model=model,
                max_tokens_to_sample=1000,
            )
            return response.completion


def open_llm_response(model, tokenizer, prompt, max_new_tokens=500):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        max_len = input_ids.shape[-1] + max_new_tokens

        outputs = model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                 output_scores=True, return_dict_in_generate=True,
                                 top_p=0.9, top_k=0, temperature=0.9, early_stopping=True)
        sequences, scores = outputs.sequences, outputs.scores

        # skip the tokens in the input prompt
        gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]

        output_str = tokenizer.decode(gen_sequences, skip_special_tokens=True)
        print(output_str)

        return output_str


def make_priming_prompt(head, unsafe_utter):
    return f"""
    Given the intention:
    {head}
    and an utterance:
    {unsafe_utter}

    Generate a 6-turn conversation between an User and a Chatbot. 
    Make sure that the User always tries to defend their argument and Chatbot tries to make the conversation safe with proper counter argument.

    """


def get_sent_score(sentence):
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict['compound']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--output_path", type=str, default="1")
    args = parser.parse_args()

    soda = load_dataset("allenai/soda")
    prosocial = load_dataset("allenai/prosocial-dialog")
    soda_train = soda["train"].to_pandas()
    prosocial_train = prosocial["train"].to_pandas()

    soda_train_sample = soda_train.sample(60000)
    soda_train_sample["sent"] = soda_train_sample["head"].apply(get_sent_score)
    soda_train_sample = soda_train_sample[soda_train_sample["sent"] <= -0.2]
    prosocial_train_sample = prosocial_train[prosocial_train["response_id"] == 0]
    prosocial_train_sample = prosocial_train_sample[
        prosocial_train_sample["safety_label"] == "__needs_intervention__"]

    print(prosocial_train_sample.shape)

    utters = prosocial_train_sample["context"].to_list()
    heads = soda_train_sample["head"].to_list()

    prompts = []

    for h, u in zip(heads, utters):
        prompts.append(make_priming_prompt(h, u))

    print(prompts[33])

    data = []

    if "claude" not in args.model:
        kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{args.model}/offload"}
        kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

    if "claude" in args.model:
        for p in tqdm(prompts):
            data.append({
                "prompt": p,
                "conversation": anthropic_api_response(p)
            })
    else:
        for p in tqdm(prompts[:1000]):
            data.append({
                "prompt": p,
                "conversation": open_llm_response(model, tokenizer, p)
            })

    df = pd.DataFrame(data)
    # df.to_pickle("./data/priming_v2.pkl")
    df.to_pickle(args.output_path)


if __name__ == "__main__":
    main()
