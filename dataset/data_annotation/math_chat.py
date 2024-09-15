import pandas as pd
from ast import literal_eval
from tqdm import tqdm, tqdm_notebook
import os
import anthropic
import re
import time
from pprint import pprint
from datasets import load_dataset
import argparse

tqdm.pandas()

# import spacy
# from sentence_transformers import SentenceTransformer, util

# sbert = SentenceTransformer('nli-distilroberta-base-v2')
# nlp = spacy.load('en_core_web_sm')

ant_client = anthropic.Client(
    api_key="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")

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is:"


def anthropic_api_response(prompt):
    model = "claude-2.1"
    max_tokens_to_sample = 1200
    try:
        response = ant_client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model,
            max_tokens_to_sample=max_tokens_to_sample,
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
                max_tokens_to_sample=max_tokens_to_sample,
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
                max_tokens_to_sample=max_tokens_to_sample,
            )
            return response.completion


def make_math_prompt(problem):
    return f"""
Given the following mathematical problem:

{problem}

Generate 5-turn conversation between an User and a Chatbot. The following rules should be followed:

1. The user should always introduce the mathematical problem.
2. The chatbot should help the user solve the mathematical problem.
3. The last utterance should contain the answer to the mathematical problem.
    """


def clean_conv(conv):
    utterances = re.findall(r"(User|Chatbot):\s(.*?)(?=(?=\w+::|$))",
                            conv, re.DOTALL)

    return utterances


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def main():
    mmathqa = load_dataset("meta-math/MetaMathQA")
    mmathqa = mmathqa["train"].to_pandas()
    mmathqa_sample = mmathqa[:15000]  # .sample(15000)

    print(mmathqa_sample.shape)

    problems = mmathqa_sample["query"].to_list()
    responses = mmathqa_sample["response"].to_list()

    prompts = []

    for p in problems:
        prompts.append(make_math_prompt(p))

    # print(prompts[33])

    data = []

    for i, p in tqdm(enumerate(prompts)):
        conv = anthropic_api_response(p)
        # if i < 3:
        # print(conv)
        conv = clean_conv(conv)

        answer = str(clean_answer(responses[i]))
        print(answer)

        is_correct = False
        for c in conv[-3:]:
            if answer in c[1]:
                is_correct = True

        print(is_correct)

        data.append({
            "prompt": p,
            "conversation": conv,
            "is_correct": is_correct,
            "real_answer": responses[i]
        })

        # if i < 3:
        # print({
        #     "prompt": p,
        #     "conversation": conv,
        #     "is_correct": is_correct
        # })

    df = pd.DataFrame(data)
    df.to_pickle("./data/math_chat_v1.pkl")


if __name__ == "__main__":
    main()
