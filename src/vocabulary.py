import os
import random
import unicodedata
import pandas as pd
import text_utils as tu

class Vocabulary:

    def __init__(self, data_folder, update=False, normalize_unicode=True):
        self.data_folder = data_folder
        self.update = update
        self.language_list = []
        self.database_path = os.path.join(data_folder, "db.pkl")
        self.vocabulary_path = os.path.join(data_folder, "words.csv")
        self.normalize_unicode = normalize_unicode
        self.df = pd.DataFrame()
        self.base_index = set()
        self.add_index = set()
        self.del_index = set()
        self.index = list()
        self.pointer = 0

    def set_language_list(self, language_list):
        self.language_list = language_list
        for language in language_list:
            if language not in self.df.columns:
                raise RuntimeError(f"{language} is not present in vocabulary.")

    def load_new_vocabulary(self):
        self.df = pd.read_csv(self.vocabulary_path)
        if self.normalize_unicode:
            for column in self.df.columns:
                if len(column) == 2:
                    self.df[column] = self.df[column].apply(lambda x: unicodedata.normalize("NFC", x))

    def load_database(self):
        self.df = pd.read_pickle(self.database_path)

    def save_database(self):
        self.df.to_pickle(self.database_path)

    def reset_scores(self):
        raise NotImplemented(f"Not implemented yet...")

    def shuffle(self):
        self.base_index = set(self.df.index.tolist())
        for i in self.del_index:
            if i in self.add_index:
                self.add_index.remove(i)
            if i in self.base_index:
                self.base_index.remove(i)
        self.index = list(self.base_index) + list(self.add_index)
        random.shuffle(self.index)
        self.pointer = 0

    def sample(self):
        if self.pointer < len(self.index):
            index = self.index[self.pointer]
            self.pointer += 1
        else:
            self.shuffle()
            self.pointer = 0
            index = self.index[self.pointer]
        return self.df.loc[index]

