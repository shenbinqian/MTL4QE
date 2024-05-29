# -*- coding: utf-8 -*-

import torch
from transformers import XLMRobertaModel
from utils import tokenize, split_for_train, load_train_dev
import numpy as np


model_name = "xlm-roberta-large"
max_length = 200


model = XLMRobertaModel.from_pretrained(model_name)

# Prepare your dataset and create data loaders
train_df, eval_df, _ = split_for_train()

train_input_ids, train_attention_masks = tokenize(train_df, model_name=model_name, max_length=max_length, use_MT=True)

eval_input_ids, eval_attention_masks = tokenize(eval_df, model_name=model_name, max_length=max_length, use_MT=True)

with torch.no_grad():
    print("Computing embeddings...")
    train_embeddings = model(torch.tensor(train_input_ids).to(torch.int64), torch.tensor(train_attention_masks).to(torch.int64))[0].numpy()
    eval_embeddings = model(torch.tensor(eval_input_ids).to(torch.int64), torch.tensor(eval_attention_masks).to(torch.int64))[0].numpy()

np.save("train_embeddings_subset_hadqaet.npy", train_embeddings)
np.save("eval_embeddings_subset_hadqaet.npy", eval_embeddings)
print("All Done!")