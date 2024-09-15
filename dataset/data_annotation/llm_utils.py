import pandas as pd
from ast import literal_eval
from tqdm import tqdm, tqdm_notebook
import os
import anthropic
import re
import time

old_key = "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"
uniwiz_key = "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"
ant_client = anthropic.Client(
    api_key=uniwiz_key)


def anthropic_api_response(prompt, model=None):
    if model is None:
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
