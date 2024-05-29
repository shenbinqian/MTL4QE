
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForSequenceClassification, AdamW, TrainingArguments, Trainer
from utils import initialize_optimizer, scheduler, tokenize, convert2dataset, convert_labels, split_for_train, load_train_dev
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd
from torch import nn


class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Load the tokenizer and model

model_name = "xlm-roberta-large" #hfl/chinese-roberta-wwm-ext
max_length = 200
batch_size = 8
num_epochs = 15
learning_rate = 2e-3

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
optimizer = initialize_optimizer(model, lr=learning_rate, eps=1e-8, weight_decay=0.01)

# Prepare your dataset and create data loaders
train_df, eval_df = load_train_dev()

class_counts = train_df.labels.value_counts().values
class_weights = 1 / class_counts
class_weights /= class_weights.sum()

train_input_ids, train_attention_masks = tokenize(train_df, model_name=model_name, max_length=max_length, use_MT=True)
train_labels = convert_labels(train_df.labels.values)
train_dataset = convert2dataset(train_input_ids, train_attention_masks, train_labels)

eval_input_ids, eval_attention_masks = tokenize(eval_df, model_name=model_name, max_length=max_length, use_MT=True)
eval_labels = convert_labels(eval_df.labels.values)
eval_dataset = convert2dataset(eval_input_ids, eval_attention_masks, eval_labels)

schedule = scheduler(optimizer, num_epochs, DataLoader(train_dataset, batch_size=batch_size), num_warmup_steps=0)

training_args = TrainingArguments(output_dir="./results",
                                logging_strategy="epoch",
                                evaluation_strategy="epoch",
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                num_train_epochs=num_epochs,
                                save_total_limit = 2,
                                save_strategy = 'no',
                                load_best_model_at_end=False
                                )


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, schedule),
    class_weights=torch.tensor(class_weights).to(torch.float32).to("cuda")
)

trainer.train()

eval_preds = trainer.predict(eval_dataset)

print(classification_report(eval_labels, eval_preds.predictions.argmax(-1)))

preds_df = pd.DataFrame({"preds": eval_preds.predictions.argmax(-1), "labels": eval_labels})
preds_df.to_csv("preds.csv", index=False)

