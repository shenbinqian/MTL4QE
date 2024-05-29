# Fine-tuning Emotion Classifiers


## Data

The HADQAET dataset is in train.csv and val.csv (used as test set) files. The MQM subset is in MQM_subset.xlsx. Relevant functions in utils.py can be called for data processing before fine-tuning.

## Fine-tuning with pre-trained XLM models

Run classifier.py to fine-tune pre-trained XLM models for emotion classification. Hyperparameters can be selected within the file.


## Train statistical classifiers with XLM embeddings

To train statistical classifiers, run save_embeddings.py first to get the embeddings from XLM-RoBERTa-large and save them into a local numpy file. Then run train.py to train statistical models such as support vector machine, random forest and ridge classifiers. Hyperparameters can be selected within the file. 

