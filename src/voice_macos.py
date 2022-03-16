import os
from voice import Voice

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
            raise RuntimeError(f"Hint: Download voice {self.voice[language]} in System Preferences -> Accessibility -> Spoken Content.")
