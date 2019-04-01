import os
import numpy as np
import pandas as pd


class WeatherAUSParser(object):
    def __init__(self):

        self.URL = "https://www.kaggle.com/jsphyg/weather-dataset-rattle-package#weatherAUS.csv"
        self.name = "weatherAUS"
        self.file_name = 'weatherAUS.csv'
        self.file_path = os.path.join("..", "..", "data", "raw", self.file_name)
        self.label_col = "RainTomorrow"
        self.X, self.y = self._parse_file()
        self.all = pd.concat([self.X, self.y], axis=1)
        self.metric = "accuracy"
        self._print_stats()

    def _parse_file(self,):
        """
            -Read csv data
            -Drop nan values
            -Keep only numeric columns
            -Split to X for features and y for labels
        """
        data = pd.read_csv(self.file_path)

        data_cleaned = data.dropna()

        X, y = data_cleaned.drop(columns=[self.label_col]), data_cleaned[self.label_col]

        # keep only numeric features
        X = X.loc[:, X.dtypes == np.float64].dropna()

        return X, y

    def save_to_csv(self):
        save_path = os.path.join("..", "..", "data", "interim", self.file_name)
        self.all.to_csv(save_path, index=False)

    def _print_stats(self):
        print("#"*30 + " Start Dataset - " + self.name + " Stats " + "#"*30)
        print("Dataset shape:", self.all.shape)
        print("Counts for each class:")
        print(self.y.value_counts())
        print("Sample of first 5 rows:")
        print(self.all.head(5))
        print("#"*30 + " End Dataset Stats " + "#"*30)


if __name__ == '__main__':

    parser = WeatherAUSParser()
    X, y = parser.X, parser.y
    parser.save_to_csv()
