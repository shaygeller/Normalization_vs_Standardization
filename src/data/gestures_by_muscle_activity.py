import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class GesturesParser(object):
    def __init__(self):

        dir_path = os.path.join("..","..","data","raw")
        file_path0 = os.path.join(dir_path, 'gestures_0.csv')
        file_path1 = os.path.join(dir_path, 'gestures_1.csv')
        file_path2 = os.path.join(dir_path, 'gestures_2.csv')
        file_path3 = os.path.join(dir_path, 'gestures_3.csv')

        X_0, y_0= self._parse_file(file_path0)
        X_1, y_1= self._parse_file(file_path1)
        X_2, y_2= self._parse_file(file_path2)
        X_3, y_3= self._parse_file(file_path3)
        X = pd.concat([X_0, X_1, X_2, X_3], axis=0)
        y = pd.concat([y_0, y_1, y_2, y_3], axis=0)
        print(X_0.shape, y_0.shape)
        print(X_1.shape, y_1.shape)
        print(X_2.shape, y_2.shape)
        print(X_3.shape, y_3.shape)
        print(X.shape, y.shape)
        X, y = shuffle(X, y, random_state=0)
        self.X, self.y = X, y
        self.all = pd.concat([X, y], axis=1)
        self._train_X, self._test_X, self._train_y, self._test_y = train_test_split(X, y)

    def _parse_file(self, file_path):
        data = pd.read_csv(file_path, header=None, names=list(range(0, 64)) +["Label"])

        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]

        return X, y

    def parse_train_set(self):
        return self._train_X, self._train_y

    def parse_test_set(self):
        return self._test_X, self._test_y

    def save_to_csv(self):
        dir_path = os.path.join("..","..","data","interim")
        self.all.to_csv(os.path.join(dir_path, __class__.__name__+'.csv'), index=False)



parser = GesturesParser()
parser.parse_train_set()
X, y = parser._train_X, parser._train_y
parser.save_to_csv()