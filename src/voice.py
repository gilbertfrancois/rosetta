import os
from typing import overload
import hashlib
import text_utils as tu

class Voice:

    def __init__(self):
        self.cache_folder = os.path.join(self.data_folder(), "voice")
        os.makedirs(self.cache_folder, exist_ok=True)

    def say(self, language, text):
        pass

    def to_mp3(self, language, text):
        pass

    def data_folder(self):
        return os.path.join(os.path.dirname(__file__), "..", "data")

    def get_filename(self, language, text, ext="mp3"):
        md5 = tu.hash(text)
        if language == "RU":
            text = tu.cyrillic2latin(text)
        else:
            text = tu.utf2ascii(text)
        text = tu.clean_str(text)
        filename = f"{language}_{text}_{md5}.{ext}"
        return filename

    def get_filepath(self, language, text, ext="mp3"):
        filename = self.get_filename(language, text, ext)
        return os.path.join(self.cache_folder, language, filename)



