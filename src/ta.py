import pytesseract
import numpy as np
import pandas as pd
import cv2 as cv
from pytesseract import Output
import input_ru as wd
import jellyfish
import os
import sys
import subprocess


np.set_printoptions(linewidth=120)
pd.set_option("display.max_rows", 100)


def experiment4():
    SCALE = 1.5
    tesseract_whitelist_de = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüÄÖÜß0123456789-,.()[]"
    )
    tesseract_whitelist_ru = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМОПРСТУФХЦЧШЩЭЮЯ0123456789-,.()[]"
    tesseract_config = (
        f"-c --psm 7 --oem 3 --load_system_dawg false --load_freq_dawg false"
    )
    tesseract_config = f"--psm 11"

    margin = 4
    image_bgr = cv.imread("../test/test-ru-de2.png")
    image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    # lines = pytesseract.image_to_string(image, output_type=Output.STRING, lang="rus")
    df = pytesseract.image_to_data(
        image_gray,
        lang="rus+deu",
        output_type=Output.DATAFRAME,
        config=tesseract_config,
    )
    df = df[df["conf"] >= 0]
    print(df.head(100))
    image_ann = image_bgr.copy()
    for index, row in df.iterrows():
        x0 = row["left"] - margin
        y0 = row["top"] - margin
        x1 = x0 + row["width"] + margin
        y1 = y0 + row["height"] + margin
        image_ann = cv.rectangle(image_ann, (x0, y0), (x1, y1), (0, 0, 200), 2)

    cv.namedWindow("res")
    cv.imshow("res", image_ann)
    cv.waitKey(0)
    cv.destroyAllWindows()


def experiment3():
    SCALE = 1.5
    tesseract_whitelist_de = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüÄÖÜß0123456789-,.()[]"
    )
    tesseract_whitelist_ru = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМОПРСТУФХЦЧШЩЭЮЯ0123456789-,.()[]"
    tesseract_config = (
        f"-c --psm 7 --oem 3 --load_system_dawg false --load_freq_dawg false"
    )
    tesseract_config_ru = (
        f"-c tessedit_char_whitelist={tesseract_whitelist_ru} --psm 11"
    )
    tesseract_config_de = (
        f"-c tessedit_char_whitelist={tesseract_whitelist_de} --psm 11"
    )

    margin = 4
    image_bgr = cv.imread("../test/test-ru-de2.png")
    image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    # lines = pytesseract.image_to_string(image, output_type=Output.STRING, lang="rus")
    df_ru = pytesseract.image_to_data(
        image_gray, lang="rus", output_type=Output.DATAFRAME, config=tesseract_config_ru
    )
    df_de = pytesseract.image_to_data(
        image_gray, lang="deu", output_type=Output.DATAFRAME, config=tesseract_config_de
    )
    df_ru = df_ru[df_ru["conf"] > 0]
    df_de = df_de[df_de["conf"] > 0]

    df_all = pd.concat([df_ru, df_de], ignore_index=True)
    df_all["purge"] = False
    bboxes = df_all.loc[:, ["left", "top", "width", "height"]].to_numpy()
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    iou_res = iou_vec(bboxes, bboxes)
    iou_res = np.triu(iou_res, k=1)
    print(iou_res)
    max_idx = np.argmax(iou_res, axis=1)
    print(max_idx)

    for i in range(iou_res.shape[0]):
        if max_idx[i] == 0 and iou_res[i, 0] < 0.01:
            continue
        conf1 = df_all.at[i, "conf"]
        conf2 = df_all.at[max_idx[i], "conf"]
        if conf1 > conf2:
            df_all.at[max_idx[i], "purge"] = True
        else:
            df_all.at[i, "purge"] = True
    dff_all = df_all[df_all["purge"] == False]
    print("-" * 80)
    #
    dff_all = dff_all.sort_values(by=["block_num"])
    #'block_num', 'par_num', 'line_num', 'word_num'
    print(dff_all)
    # print(df_all.sort_values(by=["block_num"]))

    image_ru = image_bgr.copy()
    image_de = image_bgr.copy()
    for index, row in df_ru.iterrows():
        x0 = row["left"] - margin
        y0 = row["top"] - margin
        x1 = x0 + row["width"] + margin
        y1 = y0 + row["height"] + margin
        image_ru = cv.rectangle(image_ru, (x0, y0), (x1, y1), (0, 0, 200), 2)
    for index, row in df_de.iterrows():
        x0 = row["left"] - margin
        y0 = row["top"] - margin
        x1 = x0 + row["width"] + margin
        y1 = y0 + row["height"] + margin
        image_de = cv.rectangle(image_de, (x0, y0), (x1, y1), (0, 200, 0), 2)
    image_ann = cv.addWeighted(image_ru, 0.5, image_de, 0.5, 1.0)
    # print(df_ru)
    # print(df_de)

    cv.namedWindow("res")
    cv.imshow("res", image_ann)
    cv.waitKey(0)
    cv.destroyAllWindows()


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


def iou_vec(boxes1, boxes2):
    """Computes the intersection over union (IoU) between boxes 1 and boxes 2. This is a vectorized implementation.

    Parameters
    ----------
    box1: numpy.ndarray
        Matrix of first boxes, list object with coordinates (x1, y1, x2, y2)
    box2: list of tuples
        Matrix of first boxes, list object with coordinates (x1, y1, x2, y2)

    Returns
    -------
    numpy.ndarray
        Matrix with values for the Intersection over Union, value between 0 and 1, with shape
        (n_rows_boxes1, n_rows_boxes2)
    """
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xi1 = np.maximum(x11, np.transpose(x21))
    yi1 = np.maximum(y11, np.transpose(y21))
    xi2 = np.minimum(x12, np.transpose(x22))
    yi2 = np.minimum(y12, np.transpose(y22))
    inter_area = np.maximum((xi2 - xi1 + 1), 0) * np.maximum((yi2 - yi1 + 1), 0)
    area1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = inter_area / (area1 + np.transpose(area2) - inter_area)
    return iou


if __name__ == "__main__":
    experiment4()
