import re

import os

# Import the readline module from the standard library. It automatically
# wraps stdin, solving the non working backspace with cyrilic fonts.
import readline
import subprocess
import time
from typing import Optional
import cv2 as cv
import jellyfish
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output
from structs import Context, Scan

PROMPT = "$ "
PROMPT_NORMAL = "$ "
PROMPT_INSERT = ""
PATTERN_GENDER = re.compile(r"\{.*\}", re.IGNORECASE)
PATTERN_REMARK1 = re.compile(r"\[.*\]", re.IGNORECASE)
PATTERN_REMARK2 = re.compile(r"\<.*\>", re.IGNORECASE)
SUPPORTED_SRC_LNGS = ["RU"]
SUPPORTED_LNGS = ["RU", "DE", "NL", "EN"]
OCR_BBOX_MARGIN = 4
TESSERACT_CONFIG = "-c --psm 7 --oem 3--load_system_dawg false --load_freq_dawg false"

STATUS_OK = 0
STATUS_ERROR = 1

COLUMNS = ["source_1", "source_2", "priority"] + SUPPORTED_LNGS


def read_and_predict_scan(filepath: str) -> Scan:
    scan = Scan()
    scan.filepath = filepath
    scan.image = cv.imread(scan.filepath)
    scan.image = cv.cvtColor(scan.image, cv.COLOR_BGR2GRAY)
    scan.data = pytesseract.image_to_data(
        scan.image, lang="rus", output_type=Output.DATAFRAME
    )
    scan.data = scan.data[scan.data["conf"] > 0.01]
    if len(scan.data) > 0:
        scan.success = True
    return scan


def show_scanned_word(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    crop = image[y0:y1, x0:x1]
    cv.imwrite("crop.png", crop)
    subprocess.run(
        [
            "../bin/imgcat",
            "-H",
            "1",
            "-r",
            "crop.png",
        ],
        check=False,
    )


def compute_sim(word1, word2):
    return jellyfish.jaro_similarity(word1, word2)


def match_word_for_scan(
    row: pd.Series, src_lng: str, corpus: pd.DataFrame, image: np.ndarray
) -> str:
    # image = scan.image
    # src_lng = ctx.src_lng
    # row = scan.data.iloc[scan.cursor]

    src_lng_search = f"{src_lng}_search"
    src_lng_insert = f"{src_lng}_insert"

    scan_word = str(row["text"])
    x0 = max(row["left"] - OCR_BBOX_MARGIN, 0)
    y0 = max(row["top"] - OCR_BBOX_MARGIN, 0)
    x1 = int(x0 + row["width"] + OCR_BBOX_MARGIN)
    y1 = int(y0 + row["height"] + OCR_BBOX_MARGIN)
    corpus_t = corpus.copy()
    word = scan_word.strip()
    if len(word) == 0:
        return ""

    # corpus_t["score"] = corpus_t[f"{src_lng}_search"].apply(
    #     lambda word1: jellyfish.jaro_similarity(word1, word)
    # )
    # corpus_sorted = corpus_t.sort_values(by=["score"], ascending=False).copy()
    # corpus_sorted = corpus_sorted[[f"{src_lng}_insert", "score"]]
    # corpus_sorted = corpus_sorted[corpus_sorted["score"] > 0.5]

    search_result = search_word(word, src_lng_search, corpus, fuzzy=True)

    corpus_sorted = corpus_sorted[src_lng_insert].unique()
    corpus_sorted = pd.DataFrame(corpus_sorted, columns=[src_lng_insert])
    corpus_sorted["score"] = corpus_sorted[src_lng_insert].apply(
        lambda word1: jellyfish.jaro_similarity(word1, word)
    )
    corpus_sorted = corpus_sorted.reset_index(drop=True)
    corpus_sorted.index += 1

    corpus_sorted = corpus_sorted.loc[1:5]
    corpus_sorted.loc[0] = [word, 0]
    corpus_sorted = corpus_sorted.sort_index()

    show_scanned_word(image, x0, y0, x1, y1)
    print(corpus_sorted)
    return get_user_selected_word(corpus_sorted, src_lng)


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


def search_word(
    query: str, search_col: str, corpus: pd.DataFrame, fuzzy: bool, threshold=0.4
) -> pd.DataFrame:
    search_result = corpus.copy()
    if not fuzzy:
        search_result = search_result[
            search_result[search_col].str.contains(query, case=False)
        ]
    search_result["score"] = search_result.loc[:, [search_col]].apply(
        lambda x: jellyfish.jaro_similarity(x.lower(), query.lower())
    )
    search_result = search_result[search_result["score"] > threshold]
    search_result = search_result.sort_values(by="score", ascending=False)
    search_result = search_result.reset_index(drop=True)
    search_result.index += 1
    return search_result


def search_word_unique(
    query: str, search_col: str, corpus: pd.DataFrame, sort=False
) -> pd.DataFrame:
    search_result = search_word(query, search_col, corpus, fuzzy=True)

    search_result = search_result[[f"{search_col}_insert", "score"]]
    search_result = search_result.loc[:, [f"{search_col}_insert"]].unique()
    search_result = pd.DataFrame(search_result, columns=[f"{search_col}_insert"])
    search_result["score"] = search_result[f"{search_col}_insert"].apply(
        lambda x: jellyfish.jaro_similarity(x.lower(), query.lower())
    )
    search_result = search_result.reset_index(drop=True)
    search_result.index += 1
    return search_result


def match_word_for_lng(src_word: str, src_lng: str, dst_lng: str, corpus: pd.DataFrame):
    src_lng_search = f"{src_lng}_search"
    src_lng_insert = f"{src_lng}_insert"
    dst_lng_search = f"{dst_lng}_search"
    dst_lng_insert = f"{dst_lng}_insert"
    search_result = search_word(src_word, src_lng_search, corpus, fuzzy=False)
    print(search_result.loc[:10, [src_lng_insert, dst_lng_insert, "score"]])
    return get_user_selected_word(search_result, dst_lng)


def get_user_selected_word(search_result: pd.DataFrame, lng: str) -> str:
    """
    Shows the user a short list of best matches as search result and allows the
    user to pick a proposed candidate, override by typing a unique word (i,
    followed by a string), skip (s) or end the continuous mode (.), if
    applicable.

    Parameters
    ----------
    search_result: pd.DataFrame
        Search results for the user to pick from.
    lng: str
        Language of the word that will be selected.

    Returns
    -------
    str
        word, or instruction: s (skip) or . (end)
    """
    print(
        f"Input: {search_result.index[0]}-{search_result.index[-1]}, s (skip), . (end):"
    )
    dst_lng_insert = f"{lng}_insert"
    val = input(PROMPT_INSERT)
    dst_word = ""
    if val[0].isnumeric:
        vals = val.split(",")
        vals = [int(val.strip()) for val in vals]
        rows = [search_result.loc[val] for val in vals]
        dst_words = [row[dst_lng_insert] for row in rows]
        dst_word = ", ".join(dst_words)
    elif val == "i":
        print(f"Input translation for {lng}:")
        dst_word = input(PROMPT_INSERT)
    elif val == "s":
        dst_word = "SKIP"
    elif val in ("r", ""):
        dst_word = "REPEAT"
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
    """
    Drops lines from the DataFrame that contains a NaN value.

    Parameters
    ----------
    df: pd.DataFrame
        Corpus file as pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned corpus file.
    """
    df = df.dropna(subset=df.columns[:2])
    return df


def exec_cmd_print_help() -> int:
    """
    Prints the available commands on screen.

    Returns
    -------
    bool
        Return status, ok or error.
    """
    print("--- Help commands")
    print("h              Help")
    print("s              Set source language")
    print("rs [filename]  Read scan [filename]")
    print("a              Append row")
    print("as             Append row from scan")
    print("p [nr]         Print line(s)")
    print("w              Write word list")
    print("c              Enable continuous append mode, enter . to exit")
    print("q              Quit")
    print("---")
    return STATUS_OK


def exec_cmd_read_scan(cmd: Context) -> Scan:
    """
    Read the scanned page as an image from disk and performs OCR inference. The
    results are stored in a Pandas DataFrame, including the recognised text,
    bounding boxes of the text location on paper, prediction confidence and more.

    Parameters
    ----------
    ctx: Context
        Global state

    Returns
    -------
    Scan
        Return the scan object.
    """
    if len(cmd) > 2 and cmd[:2] == "rs":
        tokens = cmd.split()
        if len(tokens) > 1:
            filepath = tokens[1]
            if not os.path.exists(filepath):
                print(f"File {filepath} does not exist.")
            else:
                return read_and_predict_scan(filepath)
    return Scan()


def exec_cmd_append(ctx: Context) -> int:
    """
    Manually append a row in Rosetta for source and all destination languages.

    Parameters
    ----------
    ctx: Context
        Global state

    Returns
    -------
    bool
        Return status, ok or error.
    """
    ctx.prev_cmd = ctx.cmd
    if ctx.src_lng not in SUPPORTED_SRC_LNGS:
        print("Error: Source language is not set.")
    else:
        src_word = input(PROMPT_INSERT)
        if src_word == ".":
            ctx.mode_continuous = False
            ctx.prev_cmd = ""
            return STATUS_OK
        row = match_word_for_lngs(src_word, ctx.src_lng, ctx.dst_lngs, corpora)
        if row is not None:
            row = {**dst_std_columns, **row}
            ctx.rosetta = pd.concat(
                [ctx.rosetta, pd.DataFrame([row])], ignore_index=True
            )
            print(ctx.rosetta.iloc[-1:])
        else:
            print("Cancelled.")
    return STATUS_OK


def exec_cmd_quit() -> bool:
    """
    Quit the program.

    Returns
    -------
    bool
        Set is_running to False if the user decides to quit.
    """
    answer = input(PROMPT_INSERT)
    if len(answer) > 0 and answer[0].lower() == "y":
        return False
    return True


def exec_cmd_write(ctx) -> int:
    """
    Write all rows of the user input from Rosetta to disk as a pandas DataFrame.

    Parameters
    ----------
    ctx: Context
        Global state

    Returns
    -------
    bool
        Return status, ok or error.
    """
    filepath = f"../data/{str(int(time.time()))}_words.csv"
    ctx.rosetta.to_csv(filepath)
    print(f"Saved {filepath}")
    return STATUS_OK


def exec_cmd_print_rosetta(ctx) -> int:
    """
    Print the requested lines from rosetta.

    Parameters
    ----------
    ctx: Context
        Global state

    Returns
    -------
    bool
        Return status, ok or error.
    """
    # No line numbers given, print all
    if len(ctx.cmd) == 1:
        print(ctx.rosetta)
        return STATUS_OK
    # Line nr(s) is/are given
    ctx.cmd, line_nrs = ctx.cmd.split()
    line_nrs = [int(nr) for nr in line_nrs.split(",")]
    # Check for valid range
    for nr in line_nrs:
        if nr not in ctx.rosetta.index:
            print("Error: Index out of range.")
            return STATUS_ERROR
    # Print the requested line numbers
    if len(line_nrs) == 1:
        print(ctx.rosetta.loc[line_nrs[0]])
        return STATUS_OK
    if len(line_nrs) == 2:
        print(ctx.rosetta.loc[line_nrs[0] : line_nrs[1]])
        return STATUS_OK
    return STATUS_ERROR


def exec_cmd_append_scan(ctx, scan) -> int:
    if ctx.src_lng == "":
        print("Error: Source language is not set.")
        ctx.prev_cmd = ""
        return STATUS_ERROR
    if scan.cursor < len(scan.data):
        scan_word = scan.data.iloc[scan.cursor]
        src_word = match_word_for_scan(
            scan_word, ctx.src_lng, corpora["RU_DE"], scan.image
        )
        scan.cursor += 1
        if src_word == ".":
            ctx.mode_continuous = False
            ctx.prev_cmd = ""
            return STATUS_OK
        if src_word == "SKIP":
            return STATUS_OK
        if src_word == "REPEAT":
            scan.cursor -= 1
            return STATUS_OK
        row = match_word_for_lngs(src_word, ctx.src_lng, ctx.dst_lngs, corpora)
        if row is not None:
            row = {**dst_std_columns, **row}
            ctx.rosetta = pd.concat(
                [ctx.rosetta, pd.DataFrame([row])], ignore_index=True
            )
            print(ctx.rosetta.iloc[-1])
        else:
            print("Cancelled.")
    else:
        print("Index out of range.")
    return STATUS_OK


def exec_cmd_set_src_lng(ctx: Context) -> int:
    ctx.src_lng = input(PROMPT_INSERT)
    if ctx.src_lng not in SUPPORTED_SRC_LNGS:
        print(f"Error. {ctx.src_lng} is not supported as source language.")
    else:
        print(f"Source language = {ctx.src_lng}")
    return STATUS_OK


def mainloop(corpora: dict[str, pd.DataFrame], dst_std_columns: dict[str, str]) -> None:
    ctx = Context()
    scan = Scan()
    ctx.rosetta = pd.DataFrame(columns=COLUMNS)

    skipped_scanned_words = []

    is_running = True
    while is_running:
        ctx.cmd = ""
        if not ctx.mode_continuous:
            try:
                ctx.cmd = input(PROMPT_NORMAL)
            except UnicodeDecodeError:
                print("Illegal character.")
                continue
        else:
            ctx.cmd = ctx.prev_cmd
        if ctx.cmd == "h":
            exec_cmd_print_help()
        elif ctx.cmd == "rs":
            scan = exec_cmd_read_scan(ctx.cmd)
        elif len(ctx.cmd) > 1 and ctx.cmd[:2] == "as":
            exec_cmd_append_scan(ctx, scan)
        elif ctx.cmd == "c":
            ctx.mode_continuous = True
        elif ctx.cmd == "s":
            exec_cmd_set_src_lng(ctx)
        elif ctx.cmd == "a":
            exec_cmd_append(ctx)
        elif ctx.cmd == "q":
            is_running = exec_cmd_quit()
        elif ctx.cmd == "w":
            exec_cmd_write(ctx)
        elif len(ctx.cmd) > 0 and ctx.cmd[0] == "p":
            exec_cmd_print_rosetta(ctx)


def get_corpora(data_folder: str):
    article_de = {"{m}": "der", "{f}": "die", "{n}": "das"}
    article_nl = {"{de}": "de", "{het}": "het"}
    article_dicts = {"DE": article_de, "NL": article_nl}
    df_ru_de = pd.read_csv(os.path.join(data_folder, "../data/ru-de.csv"))
    df_de_ru = pd.read_csv(os.path.join(data_folder, "../data/de-ru.csv"))
    df_ru_en = pd.read_csv(os.path.join(data_folder, "../data/ru-en.csv"))
    df_de_nl = pd.read_csv(os.path.join(data_folder, "../data/de-nl.csv"))

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
    print("Bootstrapping the system...")
    corpora = get_corpora("../data")

    dst_std_columns = {"source_1": "Ясно", "source_2": "6"}
    dst_df = pd.DataFrame()
    print("Ready.")
    mainloop(corpora, dst_std_columns)

    print("Bye.")
