
# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd


def initialize_optimizer(model, lr=2e-5, eps=1e-8, weight_decay=0.01):
    return AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)


def scheduler(optimizer, epochs, dataloader, num_warmup_steps=0):
    total_steps = len(dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    return scheduler


def tokenize(df, model_name="xlm-roberta-large", max_length=200, padding=True, truncation=True, return_tensors="pt", use_MT=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = []
    attention_masks = []
    if use_MT:
        for text, mt in zip(df.texts.values, df.mt.values):
            encoded_dict = tokenizer.encode_plus(
                text,
                mt,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=padding,
                truncation=truncation,
                return_attention_mask=True,
                return_tensors=return_tensors,
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
    else:
        for text in df.texts.values:
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=padding,
                truncation=truncation,
                return_attention_mask=True,
                return_tensors=return_tensors,
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx] if self.token_type_ids is not None else None,
            'labels': self.labels[idx] if self.labels is not None else None,
        }
        return item

    def __len__(self):
        return len(self.input_ids)

    def get_labels(self):
        return self.labels


def convert2dataset(input_ids, input_masks, labels):
    dataset = CustomDataset(input_ids=input_ids, attention_mask=input_masks, labels=labels)
    return dataset


def convert_labels(raw_labels):
    return torch.tensor(raw_labels.codes, dtype=torch.long)


def split_for_train(file_name="./MQM_subset.xlsx", test_frac=0.2, eval_frac=0.1, seed=823, order=True):
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_name)
    elif file_name.endswith(".tsv"):
        df = pd.read_csv(file_name, sep="\t")
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file_name)
    else:
        raise ValueError("File format not supported")

    try:
        #select columns and rename
        df = df[['source', 'MT', 'emotion_labels']]
        df = df.rename(columns={"source": "texts", "MT": "mt", "emotion_labels": "labels"})

    except:
        df = df[['src', 'mt', 'emotion_label']]
        df = df.rename(columns={"src": "texts", "emotion_label": "labels"})
    #drop rows with missing values and convert labels to categorical
    df.labels = pd.Categorical(df.labels)
    df.dropna(inplace=True)

    if order:
        train_df = df.head(int(len(df)*0.9))
        val_df = df.tail(int(len(df)*0.1))
        test_df = val_df

    else:
        test_df = df.sample(frac=test_frac, random_state=seed)
        test_df.reset_index(drop=True, inplace=True)
        train_df = df.drop(test_df.index)
        train_df.reset_index(drop=True, inplace=True)

        val_df = train_df.sample(frac=eval_frac, random_state=seed)
        val_df.reset_index(drop=True, inplace=True)
        train_df = train_df.drop(val_df.index)
        train_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df

def load_train_dev(train="./train.csv", dev="./val.csv", label_file="./annotated_dataset.csv"):
    train_df = pd.read_csv(train)
    dev_df = pd.read_csv(dev)
    label_df = pd.read_csv(label_file)

    train_labels = []
    dev_labels = []

    print("Loading labels...")
    for i in range(len(train_df)):
        for j in range(len(label_df)):
            if train_df["mt"][i].strip() == label_df["MT"][j].strip():
                train_labels.append(label_df["emotion_labels"][i])
                break

    for i in range(len(dev_df)):
        for j in range(len(label_df)):
            if dev_df["mt"][i].strip() == label_df["MT"][j].strip():
                dev_labels.append(label_df["emotion_labels"][i])
                break

    train_df["labels"] = train_labels
    dev_df["labels"] = dev_labels
    print("Done!")


    #select columns and rename
    train_df = train_df[['src', 'mt', 'labels']]
    train_df = train_df.rename(columns={"src": "texts"})

    dev_df = dev_df[['src', 'mt', 'labels']]
    dev_df = dev_df.rename(columns={"src": "texts"})


    #drop rows with missing values and convert labels to categorical
    train_df.labels = pd.Categorical(train_df.labels)
    dev_df.labels = pd.Categorical(dev_df.labels)


    return train_df, dev_df
