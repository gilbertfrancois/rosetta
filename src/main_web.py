import json
import os

from flask import Flask
from flask import redirect

from flask import render_template
from flask import request
from flask import session

app = Flask(__name__)
app.secret_key = "YrRyr_7ktWZjd2qutDLRV*p@3fkM6TTa"


@app.route("/")
def index():
    args = {"preposition": "xxxxxx",
            "article": "xxxx",
            "adjective": "aaaaa",
            "adjective_ending": "bb",
            "substantive": "ccc",
            "substantive_ending": "ccc",
            "case_id": "ccc",
            "case_desc": "ccc",
            "language_iso2": "ccc",
            "language_desc_en": "ccc",
            "language_desc_native": "ccc"
            }

    return render_template("setup3.html", args=args)


if __name__ == "__main__":
    app.debug = True
    host = os.environ.get("IP", "0.0.0.0")
    port = int(os.environ.get("PORT", 8890))
    app.run(host=host, port=port)
