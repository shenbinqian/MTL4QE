# Fine-tuning Emotion Classifiers


## Data

Data for HADQAET is in train.csv and val.csv (used as test set) files. Data for the MQM subset is in MQM_subset.xlsx. Functions in utils.py which deal with data processing are called respectively during fine-tuning.

## Fine-tuning with pre-trained XLM models

Run classifier.py to fine-tune pre-trained XLM models for emotion classification. Hyperparameters can be tuned within the file.


## Train statistical classifiers with XLM embeddings

To train statistical classifiers, run save_embeddings.py first to get the embeddings from XLM-RoBERTa-large and save them into a local numpy file. Then run train.py to train statistical models such as support vector machine, random forest and ridge classifiers. Hyperparameters can be tuned within the file. 

