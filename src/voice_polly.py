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
        # Fetch the speech file if it does not exist yet.
        filepath_mp3 = self.to_mp3(language, text, overwrite=False)
        os.system(f"play {filepath_mp3} >/dev/null 2>&1")

    def to_mp3(self, language, text, overwrite=False):
        filepath_mp3 = self.get_filepath(language, text, ext="mp3")
        if not os.path.exists(filepath_mp3) or overwrite:
            response = self.polly_client.synthesize_speech(VoiceId=self.voice.get(language),
                            OutputFormat='mp3',
                            SampleRate="22050",
                            Text=text,
                            Engine=self.engine[language])
            with open(filepath_mp3, "wb") as fp:
                fp.write(response['AudioStream'].read())
        return filepath_mp3
