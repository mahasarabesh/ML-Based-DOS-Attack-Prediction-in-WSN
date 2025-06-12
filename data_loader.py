import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, csv_path, has_target=True):
        self.csv_path = csv_path
        self.has_target = has_target
        self.df = None
        self.X = None
        self.y = None
        self.y_encoded = None
        self.class_names = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = self.df.columns.str.strip()
        if self.has_target:
            self.y = self.df.iloc[:, -1].copy()
            self.y_encoded, self.class_names = pd.factorize(self.y)
            self.X = self.df.iloc[:, :-1]
        else:
            self.X = self.df
            self.y = None
            self.y_encoded = None
            self.class_names = None

    def get_data(self):
        return self.X, self.y_encoded, self.class_names