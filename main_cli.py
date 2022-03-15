import pandas as pd
import unicodedata
import argparse
import random
import time
from colorama import Fore, Back, Style
import os
import sys


voice = {
    "RU": "milena",
    "EN": "kate",
    "DE": "anna",
    "NL": "claire"
}

MODE_AUDIT = 1
MODE_INTERACTIVE = 2

STATUS_OK = 0
STATUS_EXIT = 1

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--languages", type=str, required=True, help="Languages, comma seperated, e.g. EN,DE")
    parser.add_argument("-m", "--mode", type=str, required=True, help="1: audit, 2: interactive")
    parser.add_argument("-u", "--update-wordlist", type=bool, required=False, help="Reads the latest CSV.")
    parser.add_argument("-r", "--repeat", type=bool, required=False, help="Endless repeat.")
    return parser.parse_args(argv)

def word_info(word):
    return Fore.YELLOW + word + Style.RESET_ALL

def word_diff(word1, word2):
    result = ""
    for i in range(len(word1)):
        if i < len(word2):
            if word2[i] == word1[i]:
                result += Fore.GREEN + word1[i]
            else:
                result += Fore.RED + word1[i]
        else:
            result += Fore.RED + word1[i]
    result += Style.RESET_ALL
    return result

def say(word, voice):
    os.system(f"/usr/bin/say -v {voice} {word}")
    

def audit_word(df, index, language_list, voice, timeout=2):
    row = df.loc[index]
    _word = str(row[language_list[0]])
    _voice = voice[language_list[0]]
    print(f"{Fore.YELLOW}{_word:38s}", end="")
    sys.stdout.flush()
    say(row[language_list[0]], voice[language_list[0]])
    for i in range(1, len(language_list)):
        time.sleep(timeout)
        _word = str(row[language_list[i]])
        _voice = voice[language_list[i]]
        print(f"{Fore.WHITE}| {Fore.GREEN}{_word:38s}", end="")
        sys.stdout.flush()
        say(_word, _voice)
    print(f"{Style.RESET_ALL}\n", end="")
    sys.stdout.flush()

        
def interact_word(df, index, language_list, voice, timeout=2):
    row = df.loc[index]
    _word = row[language_list[0]]
    _voice = voice[language_list[0]]
    print(f"[ {language_list[0]} ] {Fore.YELLOW}{_word}{Style.RESET_ALL}")
    sys.stdout.flush()
    say(row[language_list[0]], voice[language_list[0]])

    for i in range(1, len(language_list)):
        _word = row[language_list[i]]
        _voice = voice[language_list[i]]
        is_correct = False
        while not is_correct:
            df.at[index, f"{language_list[0]}_{language_list[i]}_req_count"] += 1
            try:
                answer = input(f"[ {language_list[i]} ] ")
                answer = unicodedata.normalize("NFC", answer)
            except UnicodeDecodeError:
                print(f"[ EE ] Sorry, the backspace caused an error. Please try again.")
                answer = ""
                continue
            if answer == "quit":
                return STATUS_EXIT
            elif answer == "s":
                print(f"[ >> ] skipped")
                is_correct = True
            elif answer == _word:
                print(f"[ OK ] {word_diff(_word, answer)}")
                is_correct = True
            else:
                print(f"[ XX ] {word_diff(_word, answer)}")
                df.at[index, f"{language_list[0]}_{language_list[i]}_err_count"] += 1
                is_correct = False
            say(_word, _voice)
    return STATUS_OK


def load_df(db_path, language_list, words_path, update=False, normalize_unicode=True):
    if update:
        df = pd.read_csv(words_path)
    else:
        df = pd.read_pickle(db_path)
    for language in language_list:
        if language not in df.columns:
            raise RuntimeError(f"{language} is not present in vocabulary.")
    # @TODO remove line below, for debugging only.
    if normalize_unicode:
        import pdb; pdb.set_trace()
        for column in df.columns:
            if len(column) == 2:
                df[column] = df[column].apply(lambda x: unicodedata.normalize("NFC", x))
    print(f"Total number of words: {len(df)}.")
    df = df[df["source_1"] == "Ясно"]
    print(f"Number of words in selected set: {len(df)}.")
    return df

def add_count_column(df, column_list):
    for column in column_list:
        if column not in df.columns:
            df[column] = 0
        df[column].fillna(0, inplace=True)


def main(args):
    print("--------------------------------------------------------------------------------")
    print("- Rosetta                                                                      -")
    print("- Gilbert Francois Duivesteijn                                                 -")
    print("--------------------------------------------------------------------------------")
    print()

    mode = -1
    update = False
    repeat = args.repeat
    if repeat is None:
        repeat = False
    print(f"Repeat: {repeat}.")

    language_list = args.languages.split(",")
    if len(language_list) < 2:
        raise RuntimeError(f"Invalid language list.")
    try:
        mode = int(args.mode)
    except:
        raise RuntimeError(f"Mode option is invalid. Expected 1 or 2, actual {args.mode}.")
    if mode == -1:
        raise RuntimeError(f"Invalid mode.")
    if args.update_wordlist:
        update = True

    db_path = "data/db.pkl"
    words_path = "data/words.csv"

    # Load database
    df = load_df(db_path, language_list, words_path, update)

    # Add score columns if not present yet.
    req_count_list = [f"{language_list[0]}_{language_list[i]}_req_count" for i in range(1, len(language_list))]
    err_count_list = [f"{language_list[0]}_{language_list[i]}_err_count" for i in range(1, len(language_list))]
    add_count_column(df, req_count_list)
    add_count_column(df, err_count_list)

    df.to_pickle(db_path)

    if mode == MODE_INTERACTIVE:
        running = True
        while running:
            index_list = df.index.tolist()
            random.shuffle(index_list)
            for index in index_list:
                status = interact_word(df, index, language_list, voice)
                df.to_pickle(db_path)
                if status == STATUS_EXIT:
                    running = False
                    break
            if not repeat:
                running = False

    elif mode == MODE_AUDIT:
        running = True
        while running:
            index_list = df.index.tolist()
            random.shuffle(index_list)
            for index in index_list:
                audit_word(df, index, language_list, voice)
                time.sleep(3)
        if not repeat:
            running = False

    print("[ .. ] Bye!")


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
