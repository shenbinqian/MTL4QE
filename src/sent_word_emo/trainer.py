# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import datetime
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from src.common_utils import un_fit
from src.config import TEMP_DIRECTORY


def flat_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = np.array(labels).flatten()
    count = 0
    for pred, label in zip(preds_flat, labels_flat):
        if pred == label:
            count += 1
    return count / len(labels)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = round(elapsed, 2)

    return str(datetime.timedelta(seconds=elapsed_rounded))


def show_train_info(step, t0, train_dataloader, epoch_interval=40):
    # Progress update every 40 batches.
    if step % epoch_interval == 0 and not step == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


def backwards(loss, model, optimizer, scheduler=None):
    # Perform a backward pass to calculate the gradients.
    loss.backward()

    # Clip the norm of the gradients to 1.0.
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update parameters and take a step using the computed gradient.
    optimizer.step()

    # Update the learning rate.
    if scheduler:
        scheduler.step()

def initialize_optimizer(model, lr=2e-5, eps=1e-8, weight_decay=0.01):
    return AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)


def scheduler(optimizer, epochs, dataloader, num_warmup_steps=0):
    total_steps = len(dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    return scheduler


def save_model(model, optimizer, model_path="./model.pt"):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)


class Trainer:
    def __init__(self):
        pass

    def _validate(self, dev_dataloader, model, device, loss_type, statsRecord):
        print("")
        print("Running Validation...")
        statsRecord.write("Running Validation...\n")

        t0 = time.time()

        model.eval()

        # Tracking variables
        total_eval_loss = 0

        sent_predictions = []
        sent_scores = []
        word_labels = []
        word_predictions = []
        emo_predictions = []
        emo_labels = []

        for batch in dev_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_sent_score = batch[2].to(device)
            b_word_label = batch[3].to(device)
            b_emo_label = batch[4].to(device)
            b_segment_ids = batch[5].to(device)

            with torch.no_grad():
            
                loss, sent_out, word_out, emo_out = model(b_input_ids, b_input_mask, b_sent_score, b_word_label, b_emo_label, b_segment_ids, loss_type=loss_type, evaluate=True)

            # Accumulate the validation loss.
            total_eval_loss += loss

            # Move logits and labels to CPU
            sent_out = sent_out.detach().cpu().numpy()
            word_out = word_out.detach().cpu().numpy()
            emo_out = emo_out.detach().cpu().numpy()
            b_sent_score = b_sent_score.to('cpu').numpy()
            b_word_label = b_word_label.to('cpu').numpy()
            b_emo_label = b_emo_label.to('cpu').numpy()

            for sent in sent_out:
                sent_predictions.append(sent)

            for score in b_sent_score:
                sent_scores.append(score)

            word_preds = np.argmax(word_out, axis=-1).squeeze()
            for sequence, line_pred in zip(b_word_label, word_preds):
                for word, pred in zip(sequence, line_pred):
                    word_predictions.append(pred)
                    word_labels.append(word)

            for emo in emo_out:
                emo_predictions.append(emo)
            for label in b_emo_label:
                emo_labels.append(label)

        # metrics for sentence-level regression
        total_p_correlation = pearsonr(np.array(sent_predictions).squeeze(), np.array(sent_scores).squeeze())
        total_s_correlation = spearmanr(np.array(sent_predictions).squeeze(), np.array(sent_scores).squeeze())

        # metrics for word-level classification
        total_eval_accuracy = flat_accuracy(word_predictions, word_labels)
        total_f1 = f1_score(word_labels, word_predictions, average='macro')
        total_precision = precision_score(word_labels, word_predictions, average='macro', zero_division=0)
        total_recall = recall_score(word_labels, word_predictions, average='macro')

        # metrics for emotion classification
        total_emo_accuracy = flat_accuracy(emo_predictions, emo_labels)
        total_emo_f1 = f1_score(np.array(emo_labels).flatten(), np.argmax(emo_predictions, axis=-1).flatten(), average='macro')
        total_emo_precision = precision_score(np.array(emo_labels).flatten(), np.argmax(emo_predictions, axis=-1).flatten(), average='macro', zero_division=0)
        total_emo_recall = recall_score(np.array(emo_labels).flatten(), np.argmax(emo_predictions, axis=-1).flatten(), average='macro')

        # print the result of each batch
        print("Average Pearson Correlation for QE score: {0:.4f}".format(total_p_correlation[0]))
        statsRecord.write("Average Pearson Correlation for QE score: {0:.4f}\n".format(total_p_correlation[0]))
        print("Average Spearman Correlation for QE score: {0:.4f}".format(total_s_correlation[0]))
        statsRecord.write("Average Spearman Correlation for QE score: {0:.4f}\n".format(total_s_correlation[0]))
        
        print("Average F1 Score for word_level classification: {0:.4f}".format(total_f1))
        statsRecord.write("Average F1 Score for word_level classification: {0:.4f}\n".format(total_f1))
        print("Average Precision Score for word_level classification: {0:.4f}".format(total_precision))
        statsRecord.write("Average Precision Score for word_level classification: {0:.4f}\n".format(total_precision))
        print("Average Recall Score for word_level classification: {0:.4f}".format(total_recall))
        statsRecord.write("Average Recall Score for word_level classification: {0:.4f}\n".format(total_recall))
        print("Average Accuracy for word_level classification: {0:.4f}".format(total_eval_accuracy))
        statsRecord.write("Average Accuracy for word_level classification: {0:.4f}\n".format(total_eval_accuracy))

        print("Average F1 Score for emotion classification: {0:.4f}".format(total_emo_f1))
        statsRecord.write("Average F1 Score for emotion classification: {0:.4f}\n".format(total_emo_f1))
        print("Average Precision Score for emotion classification: {0:.4f}".format(total_emo_precision))
        statsRecord.write("Average Precision Score for emotion classification: {0:.4f}\n".format(total_emo_precision))
        print("Average Recall Score for emotion classification: {0:.4f}".format(total_emo_recall))
        statsRecord.write("Average Recall Score for emotion classification: {0:.4f}\n".format(total_emo_recall))
        print("Average Accuracy for emotion classification: {0:.4f}".format(total_emo_accuracy))
        statsRecord.write("Average Accuracy for emotion classification: {0:.4f}\n".format(total_emo_accuracy))

        # Calculate the average loss over all the batches.
        avg_val_loss = total_eval_loss / len(dev_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        statsRecord.write("  Validation Loss: {0:.4f}\n".format(avg_val_loss))
        statsRecord.write("  Validation took: {:}\n".format(validation_time))

    def train(self, train_dataloader, dev_dataloader, model, epochs, optimizer, device, loss_type, validate=True, scheduler=None, output="model.pt"):
        # record total time for the train vali process
        total_t0 = time.time()

        # open a txt file to log records
        statsRecord = open(TEMP_DIRECTORY + 'stats.txt', 'a')

        # move model to device
        model.to(device)

        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            statsRecord.write('======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # set model to be training mode
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                show_train_info(step, t0, train_dataloader)

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_sent_score = batch[2].to(device)
                b_word_label = batch[3].to(device)
                b_emo_label = batch[4].to(device)
                b_segment_ids = batch[5].to(device)

                model.zero_grad()

                # freeze the pretrained model
                '''
                for param in model.pretrained_model.parameters():
                    param.requires_grad = False
                '''
                    
                loss, _, _, _ = model(b_input_ids, b_input_mask, b_sent_score, b_word_label, b_emo_label, b_segment_ids, loss_type=loss_type)

                total_train_loss += loss
                backwards(loss, model, optimizer, scheduler=scheduler)
                '''
                # print the gradient of each parameter
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Parameter: {name}\nGradient:\n{param.grad}\n")
                '''

            # Calculate the average loss over all the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("Average training loss: {0:.4f}".format(avg_train_loss.item()))
            print("Training epoch took: {:}".format(training_time))
            statsRecord.write("Average training loss: {0:.4f}\n".format(avg_train_loss.item()))
            statsRecord.write("Training epoch took: {:}\n".format(training_time))

            if validate:
                self._validate(dev_dataloader, model, device, loss_type, statsRecord)

        statsRecord.close()

        print("")
        print("Training complete!")
        print("Total training and validation took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        if output:
            save_model(model, optimizer, model_path=TEMP_DIRECTORY + output)
            print("Model saved to %s" % output)


    def predict(self, test_dataloader, model, device, loss_type, out_name="predictions.csv", metrics_name="metrics.txt"):
        print("")
        print("Running Testing...")

        t0 = time.time()

        model.eval()

        # Tracking variables
        total_eval_loss = 0

        sent_predictions = []
        sent_scores = []
        word_labels = []
        word_predictions = []
        emo_predictions = []
        emo_labels = []

        word_pred_lines = []
        word_label_lines = []

        statsRecord = open(TEMP_DIRECTORY + metrics_name, 'w')

        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_sent_score = batch[2].to(device)
            b_word_label = batch[3].to(device)
            b_emo_label = batch[4].to(device)
            b_segment_ids = batch[5].to(device)

            with torch.no_grad():
                loss, sent_out, word_out, emo_out = model(b_input_ids, b_input_mask, b_sent_score, b_word_label, b_emo_label, b_segment_ids, loss_type=loss_type, evaluate=True)

            # Accumulate the validation loss.
            total_eval_loss += loss

            # Move logits and labels to CPU
            sent_out = sent_out.detach().cpu().numpy()
            word_out = word_out.detach().cpu().numpy()
            emo_out = emo_out.detach().cpu().numpy()
            b_sent_score = b_sent_score.to('cpu').numpy()
            b_word_label = b_word_label.to('cpu').numpy()
            b_emo_label = b_emo_label.to('cpu').numpy()


            for sent in sent_out:
                sent_predictions.append(sent)
            for score in b_sent_score:
                sent_scores.append(score)

            word_preds = np.argmax(word_out, axis=-1).squeeze()
            batch_preds = []
            batch_labels = []
            for sequence, line_pred in zip(b_word_label, word_preds):
                for word, pred in zip(sequence, line_pred):
                    word_predictions.append(pred)
                    word_labels.append(word)
                batch_preds.append("-".join([str(x) for x in line_pred]))
                batch_labels.append("-".join([str(x) for x in sequence]))
                word_pred_lines.append(batch_preds)
                word_label_lines.append(batch_labels)

            for emo in emo_out:
                emo_predictions.append(emo)
            for label in b_emo_label:
                emo_labels.append(label)

        # metrics for sentence-level regression
        total_p_correlation = pearsonr(np.array(sent_predictions).squeeze(), np.array(sent_scores).squeeze())
        total_s_correlation = spearmanr(np.array(sent_predictions).squeeze(), np.array(sent_scores).squeeze())

        # metrics for word-level classification
        total_eval_accuracy = flat_accuracy(word_predictions, word_labels)
        total_f1 = f1_score(word_labels, word_predictions, average='macro')
        total_precision = precision_score(word_labels, word_predictions, average='macro', zero_division=0)
        total_recall = recall_score(word_labels, word_predictions, average='macro')

        # metrics for emotion classification
        total_emo_accuracy = flat_accuracy(emo_predictions, emo_labels)
        total_emo_f1 = f1_score(np.array(emo_labels).flatten(), np.argmax(emo_predictions, axis=-1).flatten(), average='macro')
        total_emo_precision = precision_score(np.array(emo_labels).flatten(), np.argmax(emo_predictions, axis=-1).flatten(), average='macro', zero_division=0)
        total_emo_recall = recall_score(np.array(emo_labels).flatten(), np.argmax(emo_predictions, axis=-1).flatten(), average='macro')

        # print the result of each batch
        print("Average Pearson Correlation for QE score: {0:.4f}".format(total_p_correlation[0]))
        statsRecord.write("Average Pearson Correlation for QE score: {0:.4f}\n".format(total_p_correlation[0]))
        print("Average Spearman Correlation for QE score: {0:.4f}".format(total_s_correlation[0]))
        statsRecord.write("Average Spearman Correlation for QE score: {0:.4f}\n".format(total_s_correlation[0]))
        
        print("Average F1 Score for word_level classification: {0:.4f}".format(total_f1))
        statsRecord.write("Average F1 Score for word_level classification: {0:.4f}\n".format(total_f1))
        print("Average Precision Score for word_level classification: {0:.4f}".format(total_precision))
        statsRecord.write("Average Precision Score for word_level classification: {0:.4f}\n".format(total_precision))
        print("Average Recall Score for word_level classification: {0:.4f}".format(total_recall))
        statsRecord.write("Average Recall Score for word_level classification: {0:.4f}\n".format(total_recall))
        print("Average Accuracy for word_level classification: {0:.4f}".format(total_eval_accuracy))
        statsRecord.write("Average Accuracy for word_level classification: {0:.4f}\n".format(total_eval_accuracy))

        print("Average F1 Score for emotion classification: {0:.4f}".format(total_emo_f1))
        statsRecord.write("Average F1 Score for emotion classification: {0:.4f}\n".format(total_emo_f1))
        print("Average Precision Score for emotion classification: {0:.4f}".format(total_emo_precision))
        statsRecord.write("Average Precision Score for emotion classification: {0:.4f}\n".format(total_emo_precision))
        print("Average Recall Score for emotion classification: {0:.4f}".format(total_emo_recall))
        statsRecord.write("Average Recall Score for emotion classification: {0:.4f}\n".format(total_emo_recall))
        print("Average Accuracy for emotion classification: {0:.4f}".format(total_emo_accuracy))
        statsRecord.write("Average Accuracy for emotion classification: {0:.4f}\n".format(total_emo_accuracy))

        avg_val_loss = total_eval_loss / len(test_dataloader)
        print("  Test Loss: {0:.4f}".format(avg_val_loss))
        # Measure how long the test run took.
        test_time = format_time(time.time() - t0)
        print("  Test took: {:}".format(test_time))
        statsRecord.write("  Test took: {:}\n".format(test_time))
        statsRecord.close()

        sent_preds = [un_fit(sent.reshape(-1,1)) for sent in sent_predictions]
        df = pd.DataFrame({"sent_score": sent_preds, "word_tag_preds": word_pred_lines, "word_tag_labels": word_label_lines, "emo_preds":emo_predictions})
        df.to_csv(TEMP_DIRECTORY + out_name, index=False)

        return (total_p_correlation[0], total_s_correlation[0]), (total_f1, total_precision, total_recall), (total_emo_f1, total_emo_precision, total_emo_recall)
