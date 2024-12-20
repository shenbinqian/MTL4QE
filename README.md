# A Multi-task Learning Framework for Evaluating Machine Translation of Emotion-loaded User-generated Content
This repository contains the code and data to train multi-task learning (MTL) models to evaluate machine translation of emotion-loaded user-generated content (UGC). Our paper has been accepted by the Ninth Conference on Machine Translation (WMT24). Please find our paper at [arXiv](https://arxiv.org/abs/2410.03277) or [here](https://aclanthology.org/2024.wmt-1.113/) for more details. 


## Installation

```
git clone https://github.com/shenbinqian/MTL4QE.git
cd MTL4QE
conda create -n mtl4qe python=3.10
conda activate mtl4qe
pip install -r requirements.txt
```

## Train MTL models that combines sentence-, word-level QE and emotion classification

Open [src/config.py](https://github.com/shenbinqian/MTL4QE/blob/main/src/config.py) to change hyperparameters for different datasets, i.e., [HADQAET](https://github.com/shenbinqian/MTL4QE/tree/main/data/HADQAET) (Qian et al., 2023) or [MQM subset](https://github.com/shenbinqian/MTL4QE/tree/main/data/MQM_subset), task combinations (sent_word, sent_emo, or sent_word_emo), loss heuristics (Nash, Aligned, impartial MTL and etc.), pretrained language models and training hyperparameters such as the number of training epochs. The following code snippet is an example of changing these hyperparameters. 

```
MODEL_NAME = "facebook/xlm-v-base" # model name for multilingual pre-trained language models
COMBINATION = "sent_word_emo" # combination of sentence-, word-level QE and emotion classification

arg_config = {
    'model_name': MODEL_NAME,
    'loss_type': "nash", # loss heuristics: 'nash', 'aligned' or None for linear combination 
    'pool_type': 'MaxPool', # pooling strategy

    'max_seq_length': 200,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'num_train_epochs': 10, # number of train epochs
    'learning_rate': 2e-5
}
```

Run the following code in command line after specifying new hyperparameters. Or train with default hyparameters without changing hyperparameters for sentence- and word-level QE on HADQAET.

```
CUDA_VISIBLE_DEVICES=0 python -m src.run
```

## Fine-tune sentence-level QE models

To individually fine-tune sentnece-level QE models using TransQuest (Ranasinghe et al., 2020) and COMET (Rei et al., 2020), please [download](https://github.com/shenbinqian/MTL4QE/tree/main/data/ft_sent-level) our sentence-level HADQAET and MQM subset. Fine-tuning details can be found at the offical GitHub repositories of [TransQuest](https://github.com/TharinduDR/TransQuest) and [COMET](https://github.com/Unbabel/COMET).


## Fine-tune word-level QE models

To individually fine-tune word-level QE models using MicroTransQuest (Ranasinghe et al., 2021), please download our word-level [HADQAET](https://github.com/shenbinqian/MTL4QE/tree/main/data/HADQAET) and [MQM subset](https://github.com/shenbinqian/MTL4QE/tree/main/data/MQM_subset). Fuine-tuning details can be found at the offical GitHub repository of [MicroTransQuest](https://github.com/TharinduDR/TransQuest).


## Fine-tune emotion classification models

Please see our [instructions](https://github.com/shenbinqian/MTL4QE/tree/main/emo_classification) for fine-tuning emotion classification models.


## Citation

Shenbin Qian, Constantin Orasan, Diptesh Kanojia, and Félix Do Carmo. 2024. A Multi-task Learning Framework for Evaluating Machine Translation of Emotion-loaded User-generated Content. In *Proceedings of the Ninth Conference on Machine Translation*, pages 1140–1154, Miami, Florida, USA. Association for Computational Linguistics.


## BibTex Citation

```
@inproceedings{qian-etal-2024-multi,
    title = "A Multi-task Learning Framework for Evaluating Machine Translation of Emotion-loaded User-generated Content",
    author = "Qian, Shenbin  and
      Orasan, Constantin  and
      Kanojia, Diptesh  and
      Do Carmo, F{\'e}lix",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Ninth Conference on Machine Translation",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.wmt-1.113",
    pages = "1140--1154",
    abstract = "Machine translation (MT) of user-generated content (UGC) poses unique challenges, including handling slang, emotion, and literary devices like irony and sarcasm. Evaluating the quality of these translations is challenging as current metrics do not focus on these ubiquitous features of UGC. To address this issue, we utilize an existing emotion-related dataset that includes emotion labels and human-annotated translation errors based on Multi-dimensional Quality Metrics. We extend it with sentence-level evaluation scores and word-level labels, leading to a dataset suitable for sentence- and word-level translation evaluation and emotion classification, in a multi-task setting. We propose a new architecture to perform these tasks concurrently, with a novel combined loss function, which integrates different loss heuristics, like the Nash and Aligned losses. Our evaluation compares existing fine-tuning and multi-task learning approaches, assessing generalization with ablative experiments over multiple datasets. Our approach achieves state-of-the-art performance and we present a comprehensive analysis for MT evaluation of UGC.",
}
```


## References

Shenbin Qian, Constantin Orasan, Felix Do Carmo, Qiuliang Li, and Diptesh Kanojia. 2023. Evaluation of Chinese-English machine translation of emotion-loaded microblog texts: A human annotated dataset for the quality assessment of emotion translation. In *Proceedings of the 24th Annual Conference of the European Association for Machine Translation*, pages 125–135, Tampere, Finland. European Association for Machine Translation.

Tharindu Ranasinghe, Constantin Orasan, and Ruslan Mitkov. 2020. TransQuest: Translation Quality Estimation with Cross-lingual Transformers. In *Proceedings of the 28th International Conference on Computational Linguistics*, pages 5070–5081. International Committee on Computational Linguistics.

Tharindu Ranasinghe, Constantin Orasan, and Ruslan Mitkov. 2021. An exploratory analysis of multilingual word-level quality estimation with cross-lingual transformers. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, pages 434–440, Online. Association for Computational Linguistics.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A neural framework for MT evaluation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 2685–2702, Online. Association for Computational Linguistics.
