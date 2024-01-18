import cv2 as cv
import re
import time
import jellyfish
from typing import Optional
import pytesseract
from pytesseract import Output

# Import the readline module from the standard library. It automatically
# wraps stdin, solving the non working backspace with cyrilic fonts.
import readline
import pandas as pd

PROMPT = "$ "
PROMPT_NORMAL = "$ "
PROMPT_INSERT = ""
PATTERN_GENDER = re.compile(r"\{.*\}", re.IGNORECASE)
PATTERN_REMARK1 = re.compile(r"\[.*\]", re.IGNORECASE)
PATTERN_REMARK2 = re.compile(r"\<.*\>", re.IGNORECASE)
SUPPORTED_SRC_LNGS = ["RU"]
SUPPORTED_LNGS = ["RU", "DE", "NL", "EN"]

COLUMNS = ["source_1", "source_2", "priority"] + SUPPORTED_LNGS


def read_scan(filepath: str) -> list[str]:
    image = cv.imread(filepath)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lines = pytesseract.image_to_string(image, output_type=Output.STRING, lang="rus")
    lines = lines.split("\n")
    new_words = []
    for line in lines:
        word = line.strip()
        if len(word) == 0:
            continue
        new_words.append[word]
    return new_words


def compute_sim(word1, word2):
    return jellyfish.jaro_similarity(word1, word2)


def jaccard_similarity(word1, word2):
    list1 = list(word1)
    list2 = list(word2)
    iter = min(len(list1), len(list2))
    union = len(set(list1 + list2))
    intersection = 0
    for i in range(iter):
        if list1[i] == list2[i]:
            intersection = intersection + 1
    ji = intersection / union
    return ji


def match_word_for_lngs(
    src_word: str, src_lng: str, dst_lngs: list[str], corpora: dict[str, pd.DataFrame]
) -> Optional[dict]:
    row = {}
    for k, v in dst_std_columns.items():
        row[k] = v
    row[src_lng.upper()] = src_word
    for dst_lng in dst_lngs:
        key = f"{src_lng}_{dst_lng}"
        corpus = corpora.get(key)
        if corpus is None:
            raise ValueError(f"Dictionary for {src_lng}-{dst_lng} does not exist.")
        dst_word = match_word_for_lng(src_word, src_lng, dst_lng, corpus)
        if dst_word == ".":
            return None
        row[dst_lng.upper()] = dst_word
    return row


def search_word(src_word: str, src_lng: str, corpus: pd.DataFrame):
    search_result = corpus[corpus[src_lng].str.contains(src_word, case=False)].copy()
    search_result["sim"] = search_result[src_lng].apply(
        lambda dst_word: jellyfish.jaro_similarity(dst_word.lower(), src_word.lower())
    )
    search_result = search_result[search_result["sim"] > 0.4]
    search_result = search_result.sort_values(by="sim", ascending=False)
    search_result = search_result.reset_index()
    search_result.index += 1
    return search_result


def match_word_for_lng(src_word: str, src_lng: str, dst_lng: str, corpus: pd.DataFrame):
    src_lng_search = f"{src_lng}_search"
    src_lng_insert = f"{src_lng}_insert"
    dst_lng_search = f"{dst_lng}_search"
    dst_lng_insert = f"{dst_lng}_insert"
    search_result = search_word(src_word, src_lng_search, corpus)
    print(search_result.loc[:10, [src_lng_insert, dst_lng_insert, "sim"]])
    print(f"Choose a number or [i] for manual input:")
    val = input(PROMPT)
    dst_word = ""
    if val.isnumeric():
        val = int(val)
        row_src = search_result.loc[val]
        dst_word = row_src[dst_lng_insert]
    elif val == "i":
        print(f"Input translation for {dst_lng}:")
        dst_word = input(PROMPT)
    elif val == ".":
        dst_word = "."
    return dst_word


def count_tokens(x):
    return len(x.split())


def add_article(x: str, replace_dict: dict):
    """
    Replaces the gender token with the article in front of the word.

    Parameters
    ----------
    x: str
        Word as written in the corpus dictionary.
    replace_dict: dict
        { k:v }, where k=gender token and v = article.

    Returns
    -------
    Word with article before, if applicable.
    """

    genders = replace_dict.keys()

    try:
        tokens = x.split()
        for i, token in enumerate(tokens):
            if token in genders and i > 0:
                tokens[i - 1] = f"{replace_dict[token]} {tokens[i-1]}"
        x = " ".join(tokens)
    except:
        print(x)
        raise RuntimeError()
    return x


def remove_pattern(x, pattern):
    x = re.sub(pattern, "", x)
    x = " ".join(x.split())
    x = x.strip()
    return x


def clean_corpus(
    df: pd.DataFrame,
    lng: str,
    show_article: bool,
    hide_gender: bool,
    hide_remarks: bool,
    article_dicts: Optional[dict] = None,
):
    if show_article and article_dicts is not None:
        article_dict = article_dicts.get(lng)
        if article_dict is None:
            raise RuntimeError(f"No article dict found for language {lng}.")
        df[f"{lng}_insert"] = df[f"{lng}"].apply(lambda x: add_article(x, article_dict))
    else:
        df[f"{lng}_insert"] = df[lng]
    if hide_gender:
        df[f"{lng}_insert"] = df[f"{lng}_insert"].apply(
            lambda x: remove_pattern(x, PATTERN_GENDER)
        )
    if hide_remarks:
        df[f"{lng}_insert"] = df[f"{lng}_insert"].apply(
            lambda x: remove_pattern(x, PATTERN_REMARK1)
        )
        df[f"{lng}_insert"] = df[f"{lng}_insert"].apply(
            lambda x: remove_pattern(x, PATTERN_REMARK2)
        )

    df[f"{lng}_search"] = df[f"{lng}"].apply(
        lambda x: remove_pattern(x, PATTERN_GENDER)
    )
    df[f"{lng}_search"] = df[f"{lng}_search"].apply(
        lambda x: remove_pattern(x, PATTERN_REMARK1)
    )
    df[f"{lng}_search"] = df[f"{lng}_search"].apply(
        lambda x: remove_pattern(x, PATTERN_REMARK2)
    )
    return df


def dropna(df):
    v1 = len(df)
    df = df.dropna(subset=df.columns[:2])
    v2 = len(df)
    if v1 - v2 > 0:
        print(f"Dropped {v1-v2} lines.")
    return df


def mainloop(corpora: dict[str, pd.DataFrame], dst_std_columns: dict[str, str]) -> None:
    is_running = True
    src_lng = ""
    dst_lngs = ["DE", "EN"]
    df_words = pd.DataFrame(columns=COLUMNS)
    continuous_mode = False
    scan_line_index = 0
    scanned_words = []
    skipped_scanned_words = []

    while is_running:
        cmd = ""
        if not continuous_mode:
            try:
                cmd = input(PROMPT_NORMAL)
            except UnicodeDecodeError:
                print("Illegal character.")
        else:
            cmd = "i"
        if cmd == "h":
            print(f"--- Help commands")
            print(f"ls [filename]  Load scan [filename]")
            print(f"as             Prefill source language words from scan")
            print(f"h              Help")
            print(f"i              Insert row")
            print(f"s              Set source language")
            print(f"l [nr]         List line(s)")
            print(f"w              Write word list")
            print(f"c              Set continuous mode, enter . to exit")
            print(f"q              Quit")
            print(f"---")
        elif len(cmd) > 2 and cmd[:2] == "rs":
            tokens = cmd.split()
            if len(tokens) > 1:
                scanned_words = read_scan(tokens[1])
                scan_line_index = 0

        elif len(cmd) > 1 and cmd[:2] == "as":
            if len(scanned_words) <= scan_line_index:
                scan_word = scanned_words[scan_line_index]

        elif cmd == "c":
            continuous_mode = True
        elif cmd == "s":
            src_lng = input(PROMPT_INSERT)
            if src_lng not in SUPPORTED_SRC_LNGS:
                print(f"Error. {src_lng} is not supported as source language.")
            else:
                print(f"Source language = {src_lng}")
        elif cmd == "i":
            if src_lng not in SUPPORTED_SRC_LNGS:
                print("Error: Source language is not set.")
            else:
                src_word = input(PROMPT_INSERT)
                if src_word == ".":
                    continuous_mode = False
                    continue
                row = match_word_for_lngs(src_word, src_lng, dst_lngs, corpora)
                if row is not None:
                    row = {**dst_std_columns, **row}
                    df_words = pd.concat(
                        [df_words, pd.DataFrame([row])], ignore_index=True
                    )
                    print(df_words.iloc[-1])
                else:
                    print("Cancelled.")
        elif cmd == "q":
            print("Quit?")
            answer = input(PROMPT_INSERT)
            if len(answer) > 0 and answer[0].lower() == "y":
                is_running = False
        elif cmd == "w":
            filepath = f"../data/{str(int(time.time()))}_words.csv"
            df_words.to_csv(filepath)
            print(f"Saved {filepath}")
        elif len(cmd) > 0 and cmd[0] == "l":
            if len(cmd) == 1:
                print(df_words)
            else:
                cmd, linenr = cmd.split()
                if linenr not in df_words.index:
                    print(f"Error: Index out of range.")
                else:
                    print(df_words.loc[int(linenr)])


def get_corpora(data_folder: str):
    article_de = {"{m}": "der", "{f}": "die", "{n}": "das"}
    article_nl = {"{de}": "de", "{het}": "het"}
    article_dicts = {"DE": article_de, "NL": article_nl}
    df_ru_de = pd.read_csv("../data/ru-de.csv")
    df_de_ru = pd.read_csv("../data/de-ru.csv")
    df_ru_en = pd.read_csv("../data/ru-en.csv")
    df_de_nl = pd.read_csv("../data/de-nl.csv")

    df_ru_de = dropna(df_ru_de)
    df_ru_en = dropna(df_ru_en)
    df_de_nl = dropna(df_de_nl)

    df_ru_de = clean_corpus(df_ru_de, "RU", False, True, True)
    df_ru_de = clean_corpus(df_ru_de, "DE", True, True, True, article_dicts)

    df_ru_de = clean_corpus(df_de_ru, "RU", False, True, True)
    df_ru_de = clean_corpus(df_de_ru, "DE", True, True, True, article_dicts)

    df_ru_en = clean_corpus(df_ru_en, "RU", False, True, True)
    df_ru_en = clean_corpus(df_ru_en, "EN", False, False, True)

    df_de_nl = clean_corpus(df_de_nl, "DE", True, True, True, article_dicts)
    df_de_nl = clean_corpus(df_de_nl, "NL", True, True, True, article_dicts)
    corpora = {"RU_DE": df_ru_de, "RU_EN": df_ru_en, "DE_RU": df_de_ru}
    return corpora


if __name__ == "__main__":
    corpora = get_corpora("../data")
    dst_std_columns = {"source_1": "Ясно", "source_2": "6"}
    dst_df = pd.DataFrame()
    is_running = True
    mainloop(corpora, dst_std_columns)

    print("Bye.")
