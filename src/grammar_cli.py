import sqlite3
import random
from typing import Optional
from sqlite3 import DatabaseError

def create_connection(db_file:str) -> Optional[sqlite3.Connection]:
    conn:Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        print(sqlite3.version)
    except DatabaseError as e:
        print(e)
        if conn:
            conn.close()
    return conn

def get_preposition(
        conn:sqlite3.Connection, 
        language_iso2:str,
        case_id:int,
        with_random_index:bool=True):

    query = "SELECT * FROM preposition where language_iso2=? and case_id=?"
    cursor = conn.cursor()
    cursor.execute(query, (language_iso2, case_id,))
    rows = cursor.fetchall()
    res_list = _check_and_return_list(rows)
    if with_random_index:
        index = random.randint(0, len(res_list) - 1)
        return res_list[index]
    else:
        return res_list


def get_substantive_ending(
        conn:sqlite3.Connection, 
        language_iso2:str,
        gender_id:str,
        category_id:str,
        subjective_variant_id:int, 
        case_id:int):

    query = "SELECT * FROM substantive where language_iso2=? and gender_id=? and category_id=? and substantive_variant_id=? and case_id=?"
    cursor = conn.cursor()
    cursor.execute(query, (language_iso2, gender_id, category_id, subjective_variant_id, case_id,))
    rows = cursor.fetchall()
    return _check_and_return_one_value(rows)

def get_adjective_ending(
        conn:sqlite3.Connection, 
        language_iso2:str,
        gender_id:str,
        category_id:str,
        adjective_variant_id:int, 
        case_id:int):
    query = "SELECT * FROM adjective where language_iso2=? and gender_id=? and category_id=? and adjective_variant_id=? and case_id=?"
    cursor = conn.cursor()
    cursor.execute(query, (language_iso2, gender_id, category_id, adjective_variant_id, case_id,))
    rows = cursor.fetchall()
    return _check_and_return_one_value(rows)
 

def get_artikel(
        conn:sqlite3.Connection, 
        language_iso2:str,
        gender_id:str,
        category_id:str,
        article_variant_id:int, 
        case_id:int):
    query = "SELECT * FROM article where language_iso2=? and gender_id=? and category_id=? and article_variant_id=? and case_id=?"
    cursor = conn.cursor()
    cursor.execute(query, (language_iso2, gender_id, category_id, article_variant_id, case_id,))
    rows = cursor.fetchall()
    return _check_and_return_one_value(rows)


def make_article_combination(conn, adjective, substantive, language_iso2, gender_id, category_id, article_variant_id, adjective_variant_id, substantive_variant_id, case_id):
    article_row = get_artikel(conn, language_iso2, gender_id, category_id, article_variant_id, case_id)
    article = article_row["word"]
    adjective_row = get_adjective_ending(conn, language_iso2, gender_id, category_id, adjective_variant_id, case_id)
    adjective_ending = adjective_row["ending"]
    substantive_row = get_substantive_ending(conn, language_iso2, gender_id, category_id, substantive_variant_id, case_id) 
    substantive_ending = substantive_row["ending"]
    msg = f"{article} {adjective}{adjective_ending} {substantive}{substantive_ending}"
    print(msg)


def make_preposition_combination(conn, adjective, substantive, language_iso2, gender_id, category_id, article_variant_id, adjective_variant_id, substantive_variant_id, case_id):
    preposition_row = get_preposition(conn, language_iso2, case_id, with_random_index=True)
    preposition_word = preposition_row["word"]
    preposition_remark = f"({preposition_row['remark']})" if preposition_row['remark'] is not "" else ""
    article_row = get_artikel(conn, language_iso2, gender_id, category_id, article_variant_id, case_id)
    article = article_row["word"]
    adjective_row = get_adjective_ending(conn, language_iso2, gender_id, category_id, adjective_variant_id, case_id)
    adjective_ending = adjective_row["ending"]
    substantive_row = get_substantive_ending(conn, language_iso2, gender_id, category_id, substantive_variant_id, case_id) 
    substantive_ending = substantive_row["ending"]
    if preposition_word == "entlang":
        msg = f"case {case_id} | {article} {adjective}{adjective_ending} {substantive}{substantive_ending} {preposition_word} {preposition_remark}"
    else:
        msg = f"case {case_id} | {preposition_word} {article} {adjective}{adjective_ending} {substantive}{substantive_ending} {preposition_remark}"
    print(msg)


def _check_and_return_one_value(rows):
    if len(rows) > 1:
        print(rows)
        raise RuntimeError(f"More than one rows selected.")
    if len(rows) == 0:
        raise RuntimeError(f"No rows selected.")
    # res = [{k: row[k] for k in row.items()} for row in rows]
    res = []
    for row in rows:
        new_row = {}
        for k in row.keys():
            v = row[k]
            v = "" if v is None else v
            new_row[k] = v
        res.append(new_row)
    return res[0]


def _check_and_return_list(rows):
    if len(rows) == 0:
        raise RuntimeError(f"No rows selected.")
    res = []
    for row in rows:
        new_row = {}
        for k in row.keys():
            v = row[k]
            v = "" if v is None else v
            new_row[k] = v
        res.append(new_row)
    return res

if __name__ == "__main__":
    conn = create_connection("../data/grammar.db")
    if conn is None:
        exit(-1)
    
    language_iso2 = "de"
    category_id = "s"
    # article_variant_id = 1
    article_variant_id = 1
    adjective_variant_id = article_variant_id
    substantive_variant_id = 1

    adjective = "neu"
    substantive_list = [
            {"word": "Bahnhof", "gender_id": "m"},
            {"word": "Straße", "gender_id": "f"},
            {"word": "Büro", "gender_id": "n"}
        ]

    language_iso2 = "de"

    category_id = "s"
    for substantive_dict in substantive_list:
        substantive = substantive_dict["word"]
        gender_id = substantive_dict["gender_id"]
        for case_id in [1, 2, 3, 4]:
            make_article_combination(conn, adjective, substantive, language_iso2, gender_id, category_id, article_variant_id, adjective_variant_id, substantive_variant_id, case_id)
        print()

    category_id = "p"
    for substantive_dict in substantive_list:
        substantive = substantive_dict["word"]
        gender_id = substantive_dict["gender_id"]
        for case_id in [1, 2, 3, 4]:
            make_article_combination(conn, adjective, substantive, language_iso2, gender_id, category_id, article_variant_id, adjective_variant_id, substantive_variant_id, case_id)
        print()

    # adjective_variant_id = article_variant_id
    # for case_id in [1, 2, 3, 4]:
    #     make_article_combination(conn, adjective, substantive, language_iso2, gender_id, category_id, article_variant_id, adjective_variant_id, substantive_variant_id, case_id)
    #

    # for _ in range(20):
    #     article_variant_id = random.randint(1, 2)
    #     adjective_variant_id = article_variant_id
    #     print("-"*80)
    #     case_id = 3
    #     make_preposition_combination(conn, adjective, substantive, language_iso2, gender_id, category_id, article_variant_id, adjective_variant_id, substantive_variant_id, case_id)
    #     case_id = 4
    #     make_preposition_combination(conn, adjective, substantive, language_iso2, gender_id, category_id, article_variant_id, adjective_variant_id, substantive_variant_id, case_id)
    conn.close()
    #
