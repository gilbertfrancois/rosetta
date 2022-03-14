import pandas as pd
import random
import time
from colorama import Fore, Back, Style
import os
import sys


voice = {}
voice["RU"] = "milena"
voice["EN"] = "kate"
voice["DE"] = "anna"

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
    

def read_out(word1, word2, voice1, voice2):
    print(f"{Fore.YELLOW}{word1:40s} {Fore.WHITE}| ", end="")
    sys.stdout.flush()
    say(word1, voice1)
    time.sleep(2)
    print(f"{Fore.GREEN}{word2:40s}{Style.RESET_ALL}", end="\n")
    say(word2, voice2)

        
def question(word1, word2):
    voice1 = voice["DE"]
    voice2 = voice["RU"]
    status = False
    print(f"{Fore.YELLOW}{word1:40s} {Fore.WHITE}| ", end="")
    say(word1, voice1)
    try:
        answer = input("[ RU ] ")
    except UnicodeDecodeError:
        print(f"[ EE ] Sorry, the backspace caused an error. Please try again.")
        answer = ""
        status = False
        return status
    
    if answer == "s":
        print(f"[ >> ] skipped")
        return True
    elif answer == word2:
        print(f"[ √  ] {word_diff(word2, answer)}")
        say(word2, voice2)
        return True
    else:
        print(f"[ X  ] {word_diff(word2, answer)}")
        say(word2, voice2)
        return False


def main():
    UPDATE = False
    MODE = 1
    db_path = "data/db.pkl"
    words_path = "data/words.csv"

    src_lng = "DE"
    dst_lng = "RU"
    request_count = f"{src_lng}_{dst_lng}_request_count"
    error_count = f"{src_lng}_{dst_lng}_error_count"

    if UPDATE:
        df = pd.read_csv(words_path)
        df = df[df["source_1"] == "Rock 'N Learn"]
        if os.path.exists(db_path):
            ans = input("Database exists. Do you want to overwrite?")
            if ans == "y":
                df.to_pickle(db_path)
            else:
                exit(0)
    else:
        df = pd.read_pickle(db_path)
    if request_count not in df.columns:
        df[request_count] = 0
        df[error_count] = 0
        df.to_pickle(db_path)

    index_list = df.index.tolist()
    random.shuffle(index_list)
    if MODE == 0:
        for index in index_list:
            row = df.loc[index]
            is_correct = False
            while not is_correct:
                is_correct = question(row[src_lng], row[dst_lng])
                df.at[index, request_count] = df.at[index, request_count] + 1
                if not is_correct:
                    df.at[index, error_count] = df.at[index, error_count] + 1
            df.to_pickle(db_path)
    elif MODE == 1:
        while True:
            index_list = df.index.tolist()
            random.shuffle(index_list)
            for index in index_list:
                row = df.loc[index]
                read_out(row[src_lng], row[dst_lng], voice[src_lng], voice[dst_lng])
                time.sleep(3)


if __name__ == "__main__":
    main()
