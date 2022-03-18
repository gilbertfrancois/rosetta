import pandas as pd
import unicodedata
import argparse
import random
import time
from colorama import Fore, Back, Style
import os
import sys
from voice import Voice
from vocabulary import Vocabulary


OS = "macos"

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

def word_yellow(word, width=0):
    return Fore.YELLOW + word.ljust(width) + Style.RESET_ALL

def word_red(word, width=0):
    return Fore.RED + word.ljust(width) + Style.RESET_ALL

def word_green(word, width=0):
    return Fore.GREEN + word.ljust(width) + Style.RESET_ALL

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

def audit_word(vocabulary, language_list, voice, timeout, repeat):
    width = 120 // len(language_list)
    width = min(width, 40)
    width = max(width, 20)
    row = vocabulary.sample(repeat=repeat)
    if row is None:
        return False
    _language = language_list[0]
    _word = str(row[language_list[0]])
    print(f"{word_yellow(_word, width)}", end="")
    sys.stdout.flush()
    voice.say(_language, _word)
    for i in range(1, len(language_list)):
        time.sleep(timeout)
        _language = language_list[i]
        _word = str(row[language_list[i]])
        print(f"|  {word_green(_word, width)}", end="")
        sys.stdout.flush()
        voice.say(_language, _word)
        vocabulary.increase_count(row.name, "audit", language_list[0], language_list[i])
    print(f"\n", end="")
    sys.stdout.flush()
    time.sleep(timeout + 1)
    return True

        
def interact_word(vocabulary, language_list, voice, repeat):
    row = vocabulary.sample()
    _language = language_list[0]
    _word = str(row[language_list[0]])
    print(f"[ {language_list[0]} ] {word_yellow(_word)}")
    sys.stdout.flush()
    voice.say(_language, _word)
    for i in range(1, len(language_list)):
        _language = language_list[i]
        _word = str(row[language_list[i]])
        is_correct = False
        while not is_correct:
            vocabulary.increase_count(row.name, "req", language_list[0], language_list[i])
            try:
                answer = input(f"[ {language_list[i]} ] ")
                answer = unicodedata.normalize("NFC", answer)
            except UnicodeDecodeError:
                print(f"[ {word_red('EE')} ] Sorry, the backspace caused an error. Please try again.")
                answer = ""
                continue
            if answer == "quit":
                return STATUS_EXIT
            elif answer == "s":
                print(f"[ >> ] skipped")
                is_correct = True
            elif answer == _word:
                print(f"[ {word_green('Ok')} ] {word_diff(_word, answer)}")
                is_correct = True
            else:
                print(f"[ {word_red('XX')} ] {word_diff(_word, answer)}")
                vocabulary.increase_count(row.name, "err", language_list[0], language_list[i])
                is_correct = False
            voice.say(_language, _word)
    return STATUS_OK


def main(args):
    print("--------------------------------------------------------------------------------")
    print("- Rosetta                                                                      -")
    print("- Gilbert Francois Duivesteijn                                                 -")
    print("--------------------------------------------------------------------------------")
    print()

    if OS == "macos":
        from voice_macos import VoiceMacOS
        voice = VoiceMacOS()
    elif OS == "polly":
        from voice_polly import VoicePolly
        voice = VoicePolly()

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
    # Prepare cache folders for voice files.
    for lng in language_list:
        os.makedirs(f"../data/voice/{lng}", exist_ok=True)

    # Load database
    vocabulary = Vocabulary("../data", update=update)
    vocabulary.set_language_list(language_list)

    if mode == MODE_INTERACTIVE:
        running = True
        while running:
            status = interact_word(vocabulary, language_list, voice, repeat=repeat)
            vocabulary.save_database()

    elif mode == MODE_AUDIT:
        running = True
        while running:
            status = audit_word(vocabulary, language_list, voice, timeout=2, repeat=repeat)
            vocabulary.save_database()
            if not status:
                running = False

    print("[ .. ] Bye!")


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
