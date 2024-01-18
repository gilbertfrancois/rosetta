import json
import requests


API_KEY = "AIzaSyCexqbV5Ueje_Mei4LLv2wH0uuc0uiHyq8"


def detect_language(query_list):
    pass


def translate(query_list, target_language, source_language=None):
    """
    Translates the query list to the target language. When the source language is not given, the system tries to
    guess the language.

    :param query_list:          List of text strings
    :param target_language:     Target language as two letter code.
    :param source_language:     Source language as two letter code. If None is given, it tries to guess the language.
    :return:                    Result as list of dictionaries: Example:
                [{'data': {'translations': [{'detectedSourceLanguage': 'ar',
                                        'translatedText': 'City of Iraq.'},
                                       {'detectedSourceLanguage': 'ar',
                                        'translatedText': 'The forces of the regime are '
                                                          'using rockets with cluster '
                                                          'warheads to shell the plains '
                                                          'surrounding the city of Daraa'},
                                       {'detectedSourceLanguage': 'nl',
                                        'translatedText': 'The Netherlands has a new government.'}]}}]
    """

    if query_list is None:
        print("[!] Error: Input text to translate is NULL.")
        return None
    if target_language is None:
        print("[!] Error: Target language is required.")
        return None

    api_url = "https://translation.googleapis.com/language/translate/v2"

    data = {
        "q": query_list,
        "target": target_language,
        "key": API_KEY,
        "format": "text",
    }
    if source_language is not None:
        data["source"] = source_language

    response = requests.get(api_url, data)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None
