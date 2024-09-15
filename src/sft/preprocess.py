import pickle
import json
import argparse


def load_dataset(path):
    with open(
            path,
            'rb') as handle:
        data = pickle.load(handle)
    return data


def preprocess(data, i):
    inp = []
    for d in data:
        conv = []

        # if i == 0:
        conv.extend(d["conv"])

        conv.extend(d["extended_conv"])

        if len(conv) % 2 == 0:
            for i in range(0, len(conv) - 1):
                human = ""
                ai = ""
                if conv[i].startswith("User:"):
                    human += "Human: "
                    human += conv[i].split("User:")[1]

                if conv[i + 1].startswith("Chatbot:"):
                    ai += "AI: "
                    try:
                        ai += conv[i + 1].split("Chatbot:")[1]
                    except:
                        print(conv[i + 1])
                        raise NotImplementedError

                if human != "" and ai != "":
                    prompt = f"""Given an utterance by a human generate an appropriate reply: 
### Human utterance:
{human}

### AI reply:
{ai}
"""

                    inp.append({"input": prompt})

    return inp


def preprocess_1(data, i):
    inp = []
    INTRO = "Below is a conversation between a user and you."
    INSTRUCT = "Instruction: Write a response appropriate to the conversation."
    window_size = 3

    for d in data:
        conv = []

        if i == 0:
            conv.extend(d["conv"])

        conv.extend(d["extended_conv"])

        if len(conv) % 2 == 0:
            for i in range(len(conv)):
                if conv[i].startswith("Chatbot:"):
                    start = i - window_size
                    if start < 0:
                        start = 0
                    context = conv[start: i]
                    response = conv[i]

                    prompt = "\n\n".join([INTRO, "\n".join(context), INSTRUCT, response])
                    inp.append({
                        "input": prompt
                    })

    return inp


def preprocess_math(data, i):
    inp = []
    INTRO = "Below is a conversation between a user and you."
    INSTRUCT = "Instruction: Write a response appropriate to the conversation."
    window_size = 3

    for d in data:
        conv = []

        if i == 0:
            conv.extend(d["conv"])

        conv.extend(d["extended_conv"])

        if len(conv) % 2 == 0:
            for i in range(len(conv)):
                if conv[i].startswith("Chatbot:"):
                    start = i - window_size
                    if start < 0:
                        start = 0
                    context = conv[start: i]
                    response = conv[i]

                    prompt = "\n\n".join([INTRO, "\n".join(context), INSTRUCT, response])
                    inp.append({
                        "input": prompt
                    })

    return inp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_paths", nargs='+', default="huggyllama/llama-7b")
    parser.add_argument("--output_path", type=str, default="1")

    args = parser.parse_args()

    op = []
    for i, p in enumerate(args.raw_dataset_paths):
        data = load_dataset(p)
        op.extend(preprocess(data, i))

    with open(args.output_path, 'wb') as handle:
        pickle.dump(op, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
