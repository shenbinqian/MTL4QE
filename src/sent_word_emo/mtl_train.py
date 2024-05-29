# -*- coding: utf-8 -*-

import os
import shutil
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.sent_word_emo.data_prep import load_and_cache_examples, read_sent_scores, read_emo_labels
from src.common_utils import fit, reader
from src.sent_word_emo.trainer import Trainer, initialize_optimizer, scheduler
from src.config import TRAIN_PATH, TRAIN_SOURCE_FILE, \
    TRAIN_SOURCE_TAGS_FILE, \
    TRAIN_TARGET_FILE, \
    TRAIN_TARGET_TAGS_FLE, MODEL_TYPE, MODEL_NAME, arg_config, TEST_PATH, TEST_SOURCE_FILE, \
    TEST_TARGET_FILE, TEMP_DIRECTORY, SEED, DEV_PATH, DEV_SOURCE_FILE, DEV_TARGET_FILE, DEV_SOURCE_TAGS_FILE, DEV_TARGET_TAGS_FLE, \
    TRAIN_SENT_SCORE_FILE, DEV_SENT_SCORE_FILE, TEST_SENT_SCORE_FILE, TRAIN_EMO_LABEL_FILE, DEV_EMO_LABEL_FILE, TEST_EMO_LABEL_FILE
from src.microtransquest.run_model import MicroTransQuestModel
from src.sent_word_emo.mtl_modules import ExtendedHead
from transformers import AutoTokenizer, AutoModelForTokenClassification


def main():

    if not os.path.exists(TEMP_DIRECTORY):
        os.makedirs(TEMP_DIRECTORY)

    raw_train_df = reader(TRAIN_PATH, TRAIN_SOURCE_FILE, TRAIN_TARGET_FILE, TRAIN_SOURCE_TAGS_FILE,
                        TRAIN_TARGET_TAGS_FLE)
    raw_dev_df = reader(DEV_PATH, DEV_SOURCE_FILE, DEV_TARGET_FILE, DEV_SOURCE_TAGS_FILE,
                        DEV_TARGET_TAGS_FLE)
    # raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)

    train_sent_scores = read_sent_scores(TRAIN_SENT_SCORE_FILE)
    dev_sent_scores = read_sent_scores(DEV_SENT_SCORE_FILE)
    train_emo_labels = read_emo_labels(TRAIN_EMO_LABEL_FILE)
    dev_emo_labels = read_emo_labels(DEV_EMO_LABEL_FILE)
    #test_sent_scores = read_sent_scores(TEST_SENT_SCORE_FILE)
    #test_emo_labels = read_emo_labels(TEST_EMO_LABEL_FILE)

    '''fit and transform scores to be in range [0,1]'''
    train_sent_scores = fit(train_sent_scores)
    dev_sent_scores = fit(dev_sent_scores)
    # test_sent_scores = fit(test_sent_scores)

    # collect metrics for each fold
    total_p_correlation = []
    total_s_correlation = []
    total_word_f1 = []
    total_word_precision = []
    total_word_recall = []
    total_emo_f1 = []
    total_emo_precision = []
    total_emo_recall = []

    args = arg_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    '''check the prediction function of microtransquest.train_model'''

    for i in range(args["n_fold"]):

        if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
            shutil.rmtree(args['output_dir'])

        if args["evaluate_during_training"]:
            validate = True
        else:
            validate = False

        tq = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, args=args)
        pretrained_model = tq.model #AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        model = ExtendedHead(pretrained_model, pool_type=args["pool_type"])

        optimizer =initialize_optimizer(model, lr=args["learning_rate"], eps=args["adam_epsilon"], weight_decay=args["weight_decay"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=args["do_lower_case"])
    
        raw_train, raw_eval = train_test_split(raw_train_df, test_size=0.1, random_state=SEED * i)
        train_scores, dev_scores = train_test_split(train_sent_scores, test_size=0.1, random_state=SEED * i)
        train_labels, dev_labels = train_test_split(train_emo_labels, test_size=0.1, random_state=SEED * i)
        train_dataloader = load_and_cache_examples(raw_train, args, tokenizer, train_scores, train_labels, training=True)
        eval_dataloader = load_and_cache_examples(raw_eval, args, tokenizer, dev_scores, dev_labels, training=False)
        test_dataloader = load_and_cache_examples(raw_dev_df, args, tokenizer, dev_sent_scores, dev_emo_labels, training=False)

        schedule = scheduler(optimizer, args["num_train_epochs"], train_dataloader, num_warmup_steps=args["warmup_steps"])

        trainer_mtl = Trainer()
        trainer_mtl.train(train_dataloader, eval_dataloader, model, args["num_train_epochs"], optimizer, device, args["loss_type"], validate=validate, scheduler=schedule, output="model.pt")

        sent, word, emo = trainer_mtl.predict(test_dataloader, model, device, args["loss_type"], out_name="fold_{}_predictions.csv".format(str(i)), metrics_name="fold_{}_metrics.txt".format(str(i)))
        
        total_p_correlation.append(sent[0])
        total_s_correlation.append(sent[1])
        total_word_f1.append(word[0])
        total_word_precision.append(word[1])
        total_word_recall.append(word[2])
        total_emo_f1.append(emo[0])
        total_emo_precision.append(emo[1])
        total_emo_recall.append(emo[2])

    print("Pearson Correlation for QE score: {}".format(np.mean(total_p_correlation)))
    print("Spearman Correlation for QE score: {}".format(np.mean(total_s_correlation)))
    print("F1 for word tag classification: {}".format(np.mean(total_word_f1)))
    print("Precision for word tag classification: {}".format(np.mean(total_word_precision)))
    print("Recall for word tag classification:: {}".format(np.mean(total_word_recall)))
    print("F1 for emotion classification: {}".format(np.mean(total_emo_f1)))
    print("Precision for emotion classification: {}".format(np.mean(total_emo_precision)))
    print("Recall for emotion classification:: {}".format(np.mean(total_emo_recall)))


if __name__ == "__main__":
    main()