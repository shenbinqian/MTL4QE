# -*- coding: utf-8 -*-

from src.config import COMBINATION


def run_main():
    if COMBINATION == "sent_word":
        from src.sent_word.mtl_train import main
        main()
    elif COMBINATION == "sent_emo":
        from src.sent_emo.mtl_train import main
        main()
    elif COMBINATION == "sent_word_emo":
        from src.sent_word_emo.mtl_train import main
        main()
    else:
        print("Please specify the combinations of tasks in 'COMBINATION' variable in the config.py as 'sent_word', 'sent_emo' or 'sent_word_emo'!")


if __name__ == "__main__":
    run_main()