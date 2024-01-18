import input_ru as wd
import readline

PROMPT = "$ "
PROMPT_NORMAL = "$ "
PROMPT_INSERT = ""
SUPPORTED_SRC_LNGS = ["RU", "DE", "EN"]
SUPPORTED_DST_LNGS = ["RU", "DE", "EN"]


def mainloop(corpora):
    is_running = True
    src_lng = ""
    dst_lng = ""

    while is_running:
        cmd = ""
        try:
            cmd = input(PROMPT_NORMAL)
        except UnicodeDecodeError:
            print("Illegal character.")
        if cmd == "h":
            print(f"--- Help commands")
            print(f"h              Help")
            print(f"i              Insert row")
            print(f"s              Set source language")
            print(f"d              Set destination language")
            print(f"x              Swap source/destination languages")
            print(f"q              Quit")
            print(f"---")
        elif cmd == "s":
            src_lng = input(PROMPT_INSERT)
            if src_lng not in SUPPORTED_SRC_LNGS:
                print(f"Error. {src_lng} is not supported as source language.")
            else:
                print(f"Source language = {src_lng}")
        elif cmd == "d":
            dst_lng = input(PROMPT_INSERT)
            if dst_lng not in SUPPORTED_DST_LNGS:
                print(f"Error. {dst_lng} is not supported as source language.")
            else:
                print(f"Destination language = {dst_lng}")
        elif cmd == "x":
            tmp = dst_lng
            dst_lng = src_lng
            src_lng = tmp
            print(f"Set {src_lng} -> {dst_lng}")
        elif cmd == "i":
            if src_lng not in SUPPORTED_SRC_LNGS:
                print("Error: Source language is not set.")
            elif dst_lng not in SUPPORTED_DST_LNGS:
                print("Error: Destination language is not set.")
            else:
                key = f"{src_lng}_{dst_lng}"
                corpus = corpora.get(key)
                if corpus is None:
                    raise ValueError(
                        f"Dictionary for {src_lng}-{dst_lng} does not exist."
                    )
                src_word = input(PROMPT_INSERT)
                src_lng_search = f"{src_lng}_search"
                src_lng_display = f"{src_lng}"
                dst_lng_display = f"{dst_lng}"
                search_result = wd.search_word(src_word, src_lng_search, corpus)
                search_result = search_result.loc[
                    :10, [src_lng_display, dst_lng_display, "sim"]
                ].copy()
                search_result = search_result.rename(
                    columns={
                        src_lng_display: src_lng,
                        dst_lng_display: dst_lng,
                    }
                )
                if search_result.empty:
                    print("No results.")
                else:
                    print(search_result)
        elif cmd == "q":
            is_running = False


if __name__ == "__main__":
    corpora = wd.get_corpora("../data")
    mainloop(corpora)
