import os
import random
import unicodedata
import pandas as pd

class Vocabulary:

    def __init__(self, data_folder, update=False, normalize_unicode=True):
        self.data_folder = data_folder
        self.update = update
        self.language_list = []
        self.database_path = os.path.join(data_folder, "db.pkl")
        self.vocabulary_path = os.path.join(data_folder, "words.csv")
        self.normalize_unicode = normalize_unicode
        self.df = pd.DataFrame()
        # All indices in the DataFrame
        self.base_index = set()
        # Flagged by user as 'difficult word, repeat more'
        self.add_index = set()
        # Fragged by user as 'easy word, please skip'
        self.del_index = set()
        # Final index list for rehearsal.
        self.index = list()
        self.pointer = 0
        if update:
            self.load_new_vocabulary()
        self.source_sets = None

    def set_language_list(self, language_list):
        self.language_list = language_list
        for language in self.language_list:
            if language not in self.df.columns:
                raise RuntimeError(f"{language} is not present in vocabulary.")

    def load_new_vocabulary(self):
        self.df = pd.read_csv(self.vocabulary_path)
        if self.normalize_unicode:
            for column in self.df.columns:
                if len(column) == 2:
                    self.df[column] = self.df[column].apply(lambda x: unicodedata.normalize("NFC", x))
        # @TODO merge with old database to preserve scores.
        # self.df = self.df[self.df["source_2"] == "2"]
        self.save_database()
        self.get_source_sets()
        self.shuffle()

    def load_database(self):
        self.df = pd.read_pickle(self.database_path)
        self.get_source_sets()
        self.shuffle()

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

    def sample(self, repeat=True):
        if self.pointer < len(self.index):
            index = self.index[self.pointer]
            self.pointer += 1
            return self.df.loc[index]
        elif repeat:
            self.shuffle()
            index = self.index[self.pointer]
            self.pointer += 1
            return self.df.loc[index]
        else:
            return None

    def increase_count(self, index, type, lng1, lng2):
        column = f"{lng1}_{lng2}_{type}_count"
        if column not in self.df.columns:
            self.add_count_column(column)
        self.df.at[index, f"{lng1}_{lng2}_{type}_count"] += 1

    def add_score_columns(self):
        req_count_list = [f"{self.language_list[0]}_{self.language_list[i]}_req_count" for i in range(1, len(self.language_list))]
        err_count_list = [f"{self.language_list[0]}_{self.language_list[i]}_err_count" for i in range(1, len(self.language_list))]
        self.add_count_column(req_count_list)
        self.add_count_column(err_count_list)

    def add_count_column(self, column):
        if column not in self.df.columns:
            self.df[column] = 0
        self.df[column].fillna(0, inplace=True)

    def get_source_sets(self):
        dfg1 = self.df.groupby(["source_1"]).size().reset_index(name="freq")
        dfg1["source_2"] = None
        dfg1 = dfg1.sort_values(["source_1"])
        dfg2 = self.df.groupby(["source_1", "source_2"]).size().reset_index(name="freq")
        dfg2 = dfg2.sort_values(["source_1", "source_2"])
        dfc= pd.concat([dfg1, dfg2]).reset_index()
        dfc["sources_label"] = ""
        for index, row in dfc.iterrows():
            label = str(row["source_1"])
            if row["source_2"] is not None:
                label += " | " + str(row["source_2"])
            dfc.at[index, "sources_label"] = label
        return dfc
