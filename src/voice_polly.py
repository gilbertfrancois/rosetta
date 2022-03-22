import os
import boto3
from voice import Voice

class VoicePolly(Voice):

    def __init__(self):
        super().__init__()
        self.voice = {
            "DE": "Vicki",
            "EN": "Amy",
            "FR": "Léa",
            "NL": "Lotte",
            "RU": "Tatyana"
         }
        self.engine = {
                "DE": "neural",
                "EN": "neural",
                "FR": "neural",
                "NL": "standard",
                "RU": "standard"
                }

        self.polly_client = boto3.Session(profile_name="ava-x-skuld", region_name='eu-west-1').client('polly')


    def say(self, language, text):
        if self.voice.get(language) is None:
            raise RuntimeError(f"No voice mapping found for language {language}.")
        filepath = self.get_filepath(language, text)
        if not os.path.exists(filepath):
            response = self.polly_client.synthesize_speech(VoiceId=self.voice.get(language),
                            OutputFormat='mp3',
                            SampleRate="22050",
                            Text=text,
                            Engine=self.engine[language])
            with open(filepath, "wb") as fp:
                fp.write(response['AudioStream'].read())

        os.system(f"play {filepath} >/dev/null 2>&1")
