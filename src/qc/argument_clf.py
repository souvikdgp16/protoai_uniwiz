import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import copy
import argparse
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, TransfoXLForSequenceClassification, RobertaForSequenceClassification, \
    AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, TrainingArguments, Trainer
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datasets import load_dataset, load_metric
from datasets import Dataset as ds, DatasetDict
from ast import literal_eval
from sklearn.metrics import confusion_matrix


class CustomDataset(Dataset):

    def __init__(self, data, maxlen=400, with_labels=True, bert_model='albert-xxlarge-v2'):

        self.data = data  # pandas dataframe
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'topic'])
        sent2 = str(self.data.loc[index, 'argument'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        input_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attention_mask = encoded_pair['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = int(self.data.loc[index, 'label'])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                    "labels": label}
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}


class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="albert-xxlarge-v2", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        hidden_size = self.bert_layer.config.hidden_size

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 2)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps = self.bert_layer(input_ids, attention_mask, token_type_ids)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.

        logits = self.cls_layer(self.dropout(cont_reps[1]))

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # (weight=class_wt)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

            return {"loss": loss, "logits": logits}
        else:
            p = torch.nn.functional.softmax(logits, dim=1)
            return {"probs": p[:, 0]}


def custom_data_collator(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
    #                                          padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
            "labels": torch.tensor(labels)}


def compute_metrics(eval_pred):
    f1 = load_metric("f1")
    acc = load_metric("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(confusion_matrix(labels, predictions))
    return {
        "f1": f1.compute(predictions=predictions, references=labels, average=None),
        "acc": acc.compute(predictions=predictions, references=labels),
        "conf_mat": confusion_matrix(labels, predictions)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="qrecc_14.3k_random_v1_annotated_bkup.csv")
    args, extra = parser.parse_known_args()

    dataset = load_dataset("ibm/argument_quality_ranking_30k", "argument_quality_ranking")
    train_df = dataset["train"].to_pandas()
    valid_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df["label"] = np.where(train_df['MACE-P'] >= 0.7, 1, 0)
    valid_df["label"] = np.where(valid_df['MACE-P'] >= 0.7, 1, 0)
    test_df["label"] = np.where(test_df['MACE-P'] >= 0.7, 1, 0)

    # dataset = DatasetDict({
    #     "train": ds.from_pandas(train_df),
    #     "validation": ds.from_pandas(valid_df),
    #     "test": ds.from_pandas(test_df),
    # })

    train_dataset = CustomDataset(train_df)
    valid_dataset = CustomDataset(valid_df)
    test_dataset = CustomDataset(test_df)

    model = SentencePairClassifier()

    tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2")

    training_args = TrainingArguments(output_dir=args.output_path,
                                      num_train_epochs=5,
                                      save_strategy='epoch',
                                      learning_rate=3e-6,
                                      load_best_model_at_end=True,
                                      metric_for_best_model='eval_loss',
                                      evaluation_strategy="epoch",
                                      per_device_train_batch_size=4,  # batch size per device during training
                                      per_device_eval_batch_size=32,
                                      save_total_limit=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda batch: custom_data_collator(batch, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    trainer.train()

    trainer.compute_metrics = compute_metrics

    res = trainer.evaluate(
        eval_dataset=test_dataset
    )

    print(res)


if __name__ == "__main__":
    main()
