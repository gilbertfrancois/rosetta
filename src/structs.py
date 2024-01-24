from typing import Optional
import pandas as pd
import numpy as np

from dataclasses import dataclass, field


@dataclass
class Scan:
    """
    Scanned page or column from a text book.

    filepath: str
        File path of the image.
    cursor: int
        Pointer to the current line/word.
    image: np.ndarray
        Image from the scanned page.
    image_size: tuple[int, int]
        Image size (width, height).
    data: pd.DataFrame
        Results from Tesseract, like recognised words, bounding boxes and more.
    success: bool
        This is set to True if the image has been processed succesfully by the OCR method.
    """

    filepath: str = ""
    cursor: int = 0
    image: Optional[np.ndarray] = None
    image_size: tuple[int, int] = (0, 0)
    data: Optional[pd.DataFrame] = None
    success: bool = False


@dataclass
class Context:
    """
    Global state.

    """

    cmd: str = ""
    prev_cmd: str = ""
    mode_continuous: bool = False
    src_lng = "RU"
    dst_lngs = ["DE", "EN"]
    rosetta = pd.DataFrame()
    corpora: Optional[dict[str, pd.DataFrame]] = None
    prefix_columns: dict[str, str] = field(default_factory=dict)
