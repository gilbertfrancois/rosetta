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



        filepath = self.get_filepath(language, text)
        if not os.path.exists(filepath):
            status = os.system(f"/usr/bin/say -v {self.voice.get(language)} {text}")
            if status != 0:
                print(f"Hint: Download voice {self.voice[language]} in System Preferences -> Accessibility -> Spoken Content.")
                exit(1)
            
        os.system(f"play {filepath} >/dev/null 2>&1")
