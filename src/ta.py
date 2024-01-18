import pytesseract
import pandas as pd
import cv2 as cv
from pytesseract import Output
import input_ru as wd
import jellyfish


if __name__ == "__main__":
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
