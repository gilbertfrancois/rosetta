import pytesseract
import pandas as pd
import cv2 as cv
from pytesseract import Output
import input_ru as wd
import jellyfish
import os
import sys
import subprocess


def experiment2():
    SCALE = 1.5
    tesseract_config = (
        f"-c --psm 7 --oem 3--load_system_dawg false --load_freq_dawg false"
    )
    margin = 4
    image = cv.imread("../test/image1.png")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # lines = pytesseract.image_to_string(image, output_type=Output.STRING, lang="rus")
    df = pytesseract.image_to_data(image, lang="rus", output_type=Output.DATAFRAME)
    df = df[df["conf"] > 0]
    iindex = 0
    for index, row in df.iterrows():
        x0 = row["left"] - margin
        y0 = row["top"] - margin
        x1 = x0 + row["width"] + margin
        y1 = y0 + row["height"] + margin

        crop = image[y0:y1, x0:x1]
        cv.imwrite("crop.png", crop)
        subprocess.run(
            [
                "../bin/imgcat",
                "-H",
                "1",
                # f"{int(round(SCALE*row['height']))}px",
                "-r",
                "crop.png",
            ]
        )
        print(df.iloc[iindex : iindex + 1])
        iindex += 1
        x = input()


def experiment1():
    corpora = wd.get_corpora("../data")
    # tesseract_config = f"-c tessedit_char_whitelist={tesseract_whitelist} --psm 7 --oem 3--load_system_dawg false --load_freq_dawg false"
    tesseract_config = (
        f"-c --psm 7 --oem 3--load_system_dawg false --load_freq_dawg false"
    )
    corpus = corpora.get("RU_DE")
    if corpus is None:
        raise KeyError("Corpus does not exist.")
    image = cv.imread("../test/image1.png")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lines = pytesseract.image_to_string(image, output_type=Output.STRING, lang="rus")
    lines = lines.split("\n")
    for index, word in enumerate(lines):
        corpus_t = corpus.copy()
        word = word.strip()
        if len(word) == 0:
            continue
        corpus_t["dist"] = corpus_t["RU_search"].apply(
            lambda word1: jellyfish.jaro_similarity(word1, word)
        )
        corpus_sorted = corpus_t.sort_values(by=["dist"], ascending=False).copy()
        corpus_sorted = corpus_sorted[["RU_insert", "dist"]]
        corpus_sorted = corpus_sorted[corpus_sorted["dist"] > 0.5]
        corpus_sorted = corpus_sorted["RU_insert"].unique()
        corpus_sorted = pd.DataFrame(corpus_sorted, columns=["RU_insert"])
        corpus_sorted["dist"] = corpus_sorted["RU_insert"].apply(
            lambda word1: jellyfish.jaro_similarity(word1, word)
        )
        corpus_sorted = corpus_sorted.reset_index(drop=True)
        corpus_sorted.index += 1
        corpus_sorted = corpus_sorted.loc[1:5]
        corpus_sorted.loc[0] = [word, 0]
        corpus_sorted = corpus_sorted.sort_index()
        print("-" * 80)
        print(corpus_sorted)
        input("press [enter] to continue.")


if __name__ == "__main__":
    experiment2()
