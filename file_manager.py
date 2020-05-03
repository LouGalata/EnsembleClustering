import os
from scipy.io import arff
import pandas as pd
import numpy as np


class FileManager:

    def load_arff(self, dataset_name):
        dataset_path = self.__get_dataset_file_path(dataset_name)
        data, meta = arff.loadarff(dataset_path)
        df = pd.DataFrame(data)
        return df, meta

    def save_csv(self, content_data_frame, csv_file_name):
        path = "./temp/"
        if not os.path.exists(path):
            os.makedirs(path)
        content_data_frame.to_csv(path + csv_file_name + '.csv' )

    def experiment_dir(self, dataset_name):
        experiment_dir = "./temp/{}".format(dataset_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        return experiment_dir

    def save_assignments(self, assignations, dataset_name, algorithm):
        assignments_results_dir = "./clusters/{}/".format(dataset_name)
        if not os.path.exists(assignments_results_dir):
            os.makedirs(assignments_results_dir)
        np.save(assignments_results_dir+algorithm, assignations)



    def __get_dataset_file_path(self, dataset_name):
        root_dir = os.path.dirname(__file__)
        datasets_dir = os.path.join(root_dir, './resources/datasets')
        return os.path.join(datasets_dir, dataset_name + '.arff')
