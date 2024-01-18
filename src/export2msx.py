import os

from vocabulary import Vocabulary

msx_map = {
        "ю": 192,
        "а": 193,
        "б": 194,
        "ц": 195,
        "д": 196,
        "е": 197,
        "ф": 198,
        "г": 199,

        "х": 200,
        "и": 201,
        "й": 202,
        "к": 203,
        "л": 204,
        "м": 205,
        "н": 206,
        "о": 207,

        "п": 208,
        "я": 209,
        "р": 210,
        "с": 211,
        "т": 212,
        "у": 213,
        "ж": 214,
        "в": 215,

        "ь": 216,
        "ы": 217,
        "з": 218,
        "ш": 219,
        "э": 220,
        "щ": 221,
        "ч": 222,
        "ъ": 223,

        "Ю": 224,
        "А": 225,
        "Б": 226,
        "Ц": 227,
        "Д": 228,
        "Е": 229,
        "Ф": 230,
        "Г": 231,

        "Х": 232,
        "И": 233,
        "Й": 234,
        "К": 235,
        "Л": 236,
        "М": 237,
        "Н": 238,
        "О": 239,

        "П": 240,
        "Я": 241,
        "Р": 242,
        "С": 243,
        "Т": 244,
        "У": 245,
        "Ж": 246,
        "В": 247,

        "Ь": 248,
        "Ы": 249,
        "З": 250,
        "Ш": 251,
        "Э": 252,
        "Щ": 253,
        "Ч": 254
        }

def encode(src):
    dst = b""
    for i in range(len(src)):
        if src[i] in "()!? -_[]?!()":
            dst += src[i].encode("latin-1")
        elif src[i] in msx_map.keys():
            # u = bytes.fromhex(hex(msx_map[src[i]])[2:])
            u = chr(msx_map[src[i]]).encode("latin-1")
            dst += u
    if len(src) != len(dst):
        print(f"WARNING: {src} != {dst}")
    return dst


def main():
    update=True
    language_list = ["EN", "RU"]
    vocabulary = Vocabulary("../data", update=update)
    vocabulary.set_language_list(language_list)
    vocabulary.get_source_sets()
    line_nr=4000
    lines = []
    while True:
        row = vocabulary.sample(repeat=False)
        if row is None:
            break
        try:
            en_enc = row["EN"].encode("ascii") 
        except:
            continue
        line = str(line_nr).encode("latin-1") + b" data " + b"\"" + en_enc + b"\"" + b"\r\n"
        lines.append(line)
        ru_enc = encode(row["RU"])
        line = str(line_nr + 1).encode("latin-1") + b" data " + b"\"" + ru_enc + b"\"" + b"\r\n"
        lines.append(line)
        line_nr += 2
    with open(os.path.join(os.path.expanduser("~/Development/git/msx/src/basic/ru.bas")), "wb") as fp:
        fp.writelines(lines)



        
        

    


if __name__ == "__main__":
    main()
