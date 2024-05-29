from multiprocessing import cpu_count

# data file path, please change HADQAET into MQM_subset or your own data folder to test our method on other datasets
TRAIN_PATH = "data/HADQAET/train/"
TRAIN_SOURCE_FILE = "source.txt"
TRAIN_SOURCE_TAGS_FILE = "source_tags.txt"
TRAIN_TARGET_FILE = "mt.txt"
TRAIN_TARGET_TAGS_FLE = "mt_tags.txt"
TRAIN_SENT_SCORE_FILE = "./data/HADQAET/train/scores.txt"
TRAIN_EMO_LABEL_FILE = "./data/HADQAET/train/train.emotion_labels"

DEV_PATH = "data/HADQAET/dev/"
DEV_SOURCE_FILE = "source.txt"
DEV_SOURCE_TAGS_FILE = "source_tags.txt"
DEV_TARGET_FILE = "mt.txt"
DEV_TARGET_TAGS_FLE = "mt_tags.txt"
DEV_SENT_SCORE_FILE = "./data/HADQAET/dev/scores.txt"
DEV_EMO_LABEL_FILE = "./data/HADQAET/dev/dev.emotion_labels"

TEST_PATH = "data/HADQAET/test/"
TEST_SOURCE_FILE = "source.txt"
TEST_TARGET_FILE = "mt.txt"
TEST_SOURCE_TAGS_FILE = "source_tags.txt"
TEST_TARGET_TAGS_FLE = "mt_tags.txt"
TEST_SENT_SCORE_FILE = "./data/HADQAET/test/scores.txt"
TEST_EMO_LABEL_FILE = "./data/HADQAET/test/test.emotion_labels"


DEV_SOURCE_TAGS_FILE_SUB = "dev_predictions_src.txt"
DEV_TARGET_TAGS_FILE_SUB = "dev_predictions_mt.txt"


SEED = 777
TEMP_DIRECTORY = "temp/data/"
GOOGLE_DRIVE = False
DRIVE_FILE_ID = None
MODEL_TYPE = "xlmroberta"
MODEL_NAME = "facebook/xlm-v-base"  # facebook/xlm-v-base
COMBINATION = "sent_word_emo" # specify the combinations of tasks as 'sent_word', 'sent_emo' or 'sent_word_emo'

arg_config = {
    'output_dir': 'temp/outputs1/',
    "best_model_dir": "temp/outputs/best_model",
    'cache_dir': 'temp/cache_dir/',

    'model_type': MODEL_TYPE,
    'model_name': MODEL_NAME,
    'loss_type': "aligned", # loss heuristics: 'nash', 'aligned', 'imtlg', 'dwa', 'rlw' or None for linear combination 
    'pool_type': 'MaxPool', # pooling strategy

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 200,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 10, #number of train epochs
    'weight_decay': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 300,
    'save_steps': 300,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": True,
    'save_model_every_epoch': False,
    'n_fold': 1,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": True,
    'evaluate_during_training_steps': 300,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,

    'regression': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    "multiprocessing_chunksize": 500,
    'dataloader_num_workers': 0,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "manual_seed": SEED,

    "add_tag": False,
    "tag": "_",

    "default_quality": "OK",
    "labels_list": ["OK", "BAD"],

    "config": {},
    "local_rank": -1,
    "encoding": None,

    "source_column": "source",
    "target_column": "target",
    "source_tags_column": "source_tags",
    "target_tags_column": "target_tags",
}
