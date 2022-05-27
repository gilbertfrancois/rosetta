import os
import shutil
from typing import overload
from voice import Voice
from pydub import AudioSegment

class VoiceMacOS(Voice):

    def __init__(self):
        super().__init__()
        self.voice = {
            "NL": "claire",
            "DE": "anna",
            "EN": "kate",
            "RU": "milena"
         }

    def say(self, language, text):
        if self.voice.get(language) is None:
            raise RuntimeError(f"No voice mapping found for language {language}.")
        status = os.system(f"/usr/bin/say -v {self.voice.get(language)} {text}")
        if status != 0:
            print(f"Hint: Download voice {self.voice[language]} in System Preferences -> Accessibility -> Spoken Content.")
            exit(1)
            
    def to_mp3(self, language, text, overwrite=False):
        filepath_aiff = self.get_filepath(language, text, ext="aiff")
        filepath_mp3 = self.get_filepath(language, text, ext="mp3")
        if not os.path.exists(filepath_mp3) or overwrite:
            status = os.system(f"/usr/bin/say -v {self.voice.get(language)} -o {filepath_aiff} {text}")
            if status != 0:
                print(f"Non zero status code: {status}")
            speech_file = AudioSegment.from_file(filepath_aiff)
            speech_file.export(filepath_mp3, format="mp3", bitrate="48k")
            os.remove(filepath_aiff)
        return filepath_mp3



