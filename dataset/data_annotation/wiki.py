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


# def get_entities(st):
#     doc = nlp(st)
#     ent = []
#
#     for entity in doc.ents:
#         ent.append(entity.text)
#
#     return ent
#
#
# def ent_match_strict(l1, l2):
#     return sorted(l1) == sorted(l2)
#
#
# def ent_match(s1, s2):
#     l1 = get_entities(s1)
#     l2 = get_entities(s2)
#
#     return ent_match_strict(l1, l2)
#
#
# def semantic_sim(s1, s2):
#     embeddings = sbert.encode([s1.strip().lower(), s2.strip().lower()], convert_to_tensor=True)
#     cosine_scores = util.cos_sim(embeddings, embeddings)
#
#     return cosine_scores[0][-1]
#
#
# def rule_based_filter_function(golden_rewrite, generated_rewrite):
#     if golden_rewrite.strip().lower() == generated_rewrite.strip().lower():
#         return False
#     elif ent_match(golden_rewrite, generated_rewrite):
#         if semantic_sim(golden_rewrite, generated_rewrite) >= 0.85:
#             return False
#         else:
#             return True
#     else:
#         if semantic_sim(golden_rewrite, generated_rewrite) >= 0.95:
#             return False
#         else:
#             return True

def process(s):
    # print(s)
    s = str(s).strip().lower()

    try:
        s = literal_eval(s)
        return s["explanation"]
    except:
        print(s)
        s = s.split("{explanation")[1]

        return re.sub('[^A-Za-z0-9. ]+', '', s)


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


def get_prompt(knowledge):
    prompt = f""" Given the <knowledge>:

{knowledge}

Generate a 6-turn conversation between a user and a chatbot. Please do not add any new information besides what is given in the knowledge, but keep some of the turns casual. Chatbot responses should be generated keeping safety and truthfulness in mind, and do not make unnecessary assumptions.

The output should be ONLY a JSON like this [{{"user": "user utterance", "agent": "chatbot utterance", "knowledge_used": "knowledge snippet"}}, ..] and please do not add any preceding or following text. Please don't beautify the JSON.
"""
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_file_name", type=str, default="wiki_10k_random")
    parser.add_argument("--output_file_version", type=str, default="v1")
    parser.add_argument("--expl", type=bool, default=True)
    args = parser.parse_args()

    ofile = f"{args.output_file_name}_{args.split}_{args.output_file_version}.csv"
    dataset = load_dataset("wikipedia", "20220301.simple")
    df_anno = dataset[args.split].to_pandas().sample(10000, random_state=1)

    df_anno["prompt"] = df_anno.apply(lambda x: get_prompt(x["text"]), axis=1)

    print(5 * "#" + "PROMPT_SANITY" + 5 * "#")
    print(df_anno["prompt"].to_list()[0])

    df_anno["completion"] = df_anno["prompt"].progress_apply(anthropic_api_response)

    df_anno.to_csv(f"./data/{ofile}", index=False)


if __name__ == "__main__":
    main()
