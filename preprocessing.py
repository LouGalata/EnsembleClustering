import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    MultiLabelBinarizer
)


class Preprocessor:
    NULL_RATIO_THRESHOLD = 0.2
    NUM_OF_BINS = 20

    def __init__(self, data):
        # Exclude the last column (ground-truth)
        self.data = data.iloc[:, :-1]
        # store the last column separately
        self.classification_data = data.iloc[:, -1]
        # Initialize auxiliary dataset feature vectors
        self.numerical_feat = None
        self.nominal_feat = None
        self.numerical_to_nominal_feat = None
        self.nominal_to_numerical_feat = None

    def preprocess(self):
        print(" * Preprocessing data...")
        self.__preprocess_numerical_data()
        self.__preprocess_nominal_data()
        self.numerical_to_nominal_feat = self.__numerical_to_nominal()
        self.nominal_to_numerical_feat = self.__nominal_to_numerical()

    def get_numerical(self):
        return pd.concat([self.numerical_feat, self.nominal_to_numerical_feat], axis=1)

    def get_classification_data(self):
        return self.classification_data.copy()

    def __get_orig_numerical_features(self):
        return self.data.select_dtypes(exclude=[np.object]).copy()

    def __get_orig_nominal_features(self):
        nom_features = self.data.select_dtypes([np.object])
        if not nom_features.empty:
            nom_features = nom_features.stack().str.decode('utf-8').unstack()
            nom_features = nom_features.replace('?', np.NaN)
        return nom_features.copy()

    def __preprocess_numerical_data(self):
        self.numerical_feat = self.__get_orig_numerical_features()
        scaler = MinMaxScaler()
        for i in self.numerical_feat:
            # Data normalization
            transposed_col = self.numerical_feat[i].values.reshape(-1, 1)
            self.numerical_feat.loc[:, i] = scaler.fit_transform(transposed_col)
            # Handle the empty values
            self.__handle_numerical_null_values(i)

    def __handle_numerical_null_values(self, i):
        length = len(self.data.index)
        null_values = self.numerical_feat[i].isnull().sum()
        null_ratio = null_values / length
        if null_ratio > self.NULL_RATIO_THRESHOLD:
            self.numerical_feat.drop([i], axis=1, inplace=True)
        else:
            self.__imputation_of_missing_numerical_values(i)

    def __imputation_of_missing_numerical_values(self, i):
        has_meaningful_values = self.numerical_feat[i].any()
        has_missing_values = self.numerical_feat[i].isnull().values.any()
        if has_meaningful_values and has_missing_values:
            mean = self.numerical_feat[i].mean()
            self.numerical_feat[i].fillna(value=mean, inplace=True)

    def __preprocess_nominal_data(self):
        self.nominal_feat = self.__get_orig_nominal_features()
        for i in self.nominal_feat:
            length = len(self.data.index)
            null_values = self.nominal_feat[i].isnull().sum()
            null_ratio = null_values / length
            if null_ratio > self.NULL_RATIO_THRESHOLD:
                self.nominal_feat.drop([i], axis=1, inplace=True)
            else:
                self.__imputation_of_missing_nominal_values(i)

    def __imputation_of_missing_nominal_values(self, i):
        has_meaningful_values = self.nominal_feat[i].any()
        has_missing_values = self.nominal_feat[i].isnull().values.any()
        if has_meaningful_values and has_missing_values:
            top = self.nominal_feat[i].describe()['top']
            self.nominal_feat[i].fillna(top, inplace=True)

    def __nominal_to_numerical(self):
        converted_features = None
        for i in self.nominal_feat:
            converted_features = self.__normalize_nominal_values(i, converted_features)
        return converted_features

    def __normalize_nominal_values(self, i, converted_features):
        transposed_col = self.nominal_feat[i].values.reshape(-1, 1)
        index = self.nominal_feat[i].index
        mlb = MultiLabelBinarizer()
        transformed_col = mlb.fit_transform(transposed_col)
        df = pd.DataFrame(transformed_col, columns=mlb.classes_, index=index)
        result = pd.concat([converted_features, df], axis=1)
        return result

    def __numerical_to_nominal(self):
        converted_features = None
        for i in self.numerical_feat:
            column = self.numerical_feat[i]
            new_converted_feat = pd.qcut(column, self.NUM_OF_BINS, duplicates='drop')
            converted_features = pd.concat([converted_features, new_converted_feat], axis=1)
        return converted_features
