# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from src.microtransquest.utils import get_examples_from_df, convert_examples_to_features
from src.microtransquest.format import prepare_data


def load_and_cache_examples(raw_df, args, tokenizer, sent_scores, emo_labels, no_cache=False, training=True):

    process_count = args["process_count"]

    if not no_cache:
        no_cache = args["no_cache"]

    data_df = prepare_data(raw_df, args)
    examples = get_examples_from_df(data_df, bbox=False)

    cached_features_file = os.path.join(
        args["cache_dir"],
        "cached_{}_{}_{}_{}".format(
            args["model_type"], args["max_seq_length"], 2, len(examples),
        ),
    )
    if not no_cache:
        os.makedirs(args["cache_dir"], exist_ok=True)

    if os.path.exists(cached_features_file) and (
            (not args["reprocess_input_data"] and not no_cache)):
        features = torch.load(cached_features_file)
    else:
        features = convert_examples_to_features(
            examples,
            args["labels_list"],
            args["max_seq_length"],
            tokenizer,
            # XLNet has a CLS token at the end
            cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            # RoBERTa uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            sep_token_extra=bool(args["model_type"] in ["roberta"]),
            # PAD on the left for XLNet
            pad_on_left=bool(args["model_type"] in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
            pad_token_label_id=CrossEntropyLoss().ignore_index,
            process_count=process_count,
            silent=args["silent"],
            use_multiprocessing=args["use_multiprocessing"],
            chunksize=args["multiprocessing_chunksize"],
        )

        if not no_cache:
            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.int64)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.int64)
        all_sent_scores = torch.tensor(sent_scores, dtype=torch.float)
        all_emo_labels = torch.tensor(emo_labels, dtype=torch.int64)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.int64)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_sent_scores, all_emo_labels, all_segment_ids)

        if training:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args["train_batch_size"],
            num_workers=args["dataloader_num_workers"],
        )

    return dataloader



def read_sent_scores(sent_scores_file):
    sent_scores = []
    with open(sent_scores_file, 'r') as f:
        for line in f:
            sent_scores.append(float(line.strip()))
    return np.array(sent_scores).reshape(-1, 1)


def read_emo_labels(emo_labels_file):
    emo_labels = []
    with open(emo_labels_file, 'r') as f:
        for line in f:
            emo_labels.append(int(line.strip()))
    return np.array(emo_labels).reshape(-1, 1)