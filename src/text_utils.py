import unicodedata
import string
import hashlib

def cyrillic2latin(text):
    text = unicodedata.normalize("NFC", text)
    _map = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E',
        'Ё': 'E', 'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K',
        'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R',
        'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'Ts',
        'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
        'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya', 
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
        'ё': 'e', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k',
        'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
        'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
        'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '',
        'э': 'e', 'ю': 'yu', 'я': 'ya' }
    res = ""
    for char in text:
        if char in _map.keys():
            res += _map[char]
        else:
            res += char
    return res 

def utf2ascii(text):
    text = text.replace("ß", "ss")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode("ascii")
    return text

def clean_str(text):
    text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.replace("\r", "")
    text = text.lower()
    return text

def hash(text):
    return hashlib.md5(text.encode()).hexdigest()


