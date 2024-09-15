import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from llm_utils import anthropic_api_response
import yake
import random
from tqdm import tqdm
import argparse
from datasets import load_dataset


def load_pickle(path):
    with open(path, 'rb') as handle:
        d = pickle.load(handle)

    return d


def tokenize_query(query):
    return query.split(" ")


def open_file(path):
    with open(path) as f:
        lines = f.readlines()

    return lines


class UniWiz:
    def __init__(self, fact_path, safety_primed_conv_path, names_path):
        self.names = open_file(names_path)
        self.fact = load_pickle(fact_path)
        self.fact_corpus = list(self.fact.keys())
        self.fact_index = self._index_fact(self.fact_corpus)
        self.primed_convs = load_pickle(safety_primed_conv_path)
        self.kw_extractor = yake.KeywordExtractor()
        self.primed_convs["conv_cleaned"] = self.primed_convs["conversation"].apply(self.clean_conv)
        self.primed_convs = self.primed_convs[self.primed_convs["conv_cleaned"].str.len() > 2]

        print("Shape of Primed corpus:")
        print(self.primed_convs.shape)

    def _index_fact(self, corpus):
        tokenized_corpus = [doc.split(" ") for doc in corpus]

        return BM25Okapi(tokenized_corpus)

    def clean_conv(self, conv):
        name = random.choice(self.names).strip('\n')
        lines = conv.split("\n")
        filtered = []
        for l in lines:
            if l.startswith("User:") or l.startswith("Chatbot:"):
                l = l.replace("PersonX", name).replace("PersonY", name)
                filtered.append(l)

        return filtered

    def get_keywords(self, text):
        keywords = self.kw_extractor.extract_keywords(text)
        filtered_kw = []

        for kw in keywords:
            if kw[0].lower().strip() not in ["user", "chatbot"]:
                filtered_kw.append(kw[0])

        return filtered_kw

    def get_prompt(self, conv, knowledge, instruction=False):
        conv = "\n\n".join(conv)
        knowledge = " ".join(knowledge)
        if instruction:
            return f"""Given the conversation between a User and Chatbot:
{conv}
, given a knowledge:
{knowledge}
 and a instruction:
{instruction}
Extend the given conversation between a User and Chatbot for 6 turns. The conversation should start with the User."""
        else:
            return f"""Given the conversation between a User and Chatbot:
{conv}
, and given a knowledge:
{knowledge}
Extend the given conversation between a User and Chatbot for 10 turns. The conversation should start with the User."""

    def orchestrate(self, k=1, n=5, kw_k=1, depth=1, inst_df=None):
        data = []
        for conv in tqdm(self.primed_convs.to_dict(orient="records")):
            keywords = self.get_keywords(conv["conversation"])

            knowledge = []
            for i in range(0, kw_k):
                key = self.fact_index.get_top_n(tokenize_query(keywords[i]), self.fact_corpus, n=n)
                kn = sorted(set(self.fact[key[k - 1]]), key=self.fact[key[k - 1]].index)
                knowledge.append(kn[:depth])

            for kn in knowledge:
                try:
                    if inst_df.shape:
                        inst = inst_df.sample(1)["instruction"].to_list()[0]
                        prompt = self.get_prompt(conv["conv_cleaned"], kn, inst)
                except:
                    print("No inst_df !!")
                    prompt = self.get_prompt(conv["conv_cleaned"], kn)

                extended_conv = anthropic_api_response(prompt)
                extended_conv_cleaned = self.clean_conv(extended_conv)

                data.append({
                    "conv": conv["conv_cleaned"],
                    "extended_conv": extended_conv_cleaned,
                    "knowledge_used": kn,
                    "n": n,
                    "k": k,
                    "kw_k": kw_k,
                    "depth": depth
                })

        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fact_path", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--safety_primed_conv_path", type=str, default="1")
    parser.add_argument("--names_path", type=str, default=27)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--kw_k", type=int, default=1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="./tfqa_result")
    parser.add_argument("--instruct_dataset", type=str)

    args = parser.parse_args()

    if args.instruct_dataset == "yahma/alpaca-cleaned":
        dataset = load_dataset("yahma/alpaca-cleaned")

        df = dataset["train"].to_pandas()
        df = df[df["input"] == '']
    else:
        df = None

    uw = UniWiz(args.fact_path,
                args.safety_primed_conv_path,
                args.names_path
                )
    data = uw.orchestrate(
        args.k,
        args.n,
        args.kw_k,
        args.depth,
        df
    )

    with open(args.output_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
