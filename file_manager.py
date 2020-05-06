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

    def save_assignments(self, assignations, dataset_name, exp_num, are_clusters_fixed, algorithm):
        if are_clusters_fixed:
            assignments_results_dir = "./clusters/{}/fixed_k/{}/".format(dataset_name, algorithm)
        else:
            assignments_results_dir = "./clusters/{}/random_k/{}/".format(dataset_name, algorithm)

        if not os.path.exists(assignments_results_dir):
            os.makedirs(assignments_results_dir)

        np.save(assignments_results_dir + 'ensembling_' + str(exp_num), assignations)

    def save_ensembling_matrices(self, dataset_name, matrix_data, matrix_name, are_clusters_fixed):
        if are_clusters_fixed == 'fixed_k':
            assignments_results_dir = "./matrices/{}/fixed_k/".format(dataset_name)
        else:
            assignments_results_dir = "./matrices/{}/random_k/".format(dataset_name)

        if not os.path.exists(assignments_results_dir):
            os.makedirs(assignments_results_dir)

        pd.DataFrame(matrix_data).to_csv(assignments_results_dir + '{}.csv'.format(matrix_name), index=False)

    def get_ensembling_matrices(self, dataset_name, are_clusters_fixed):
        if are_clusters_fixed:
            assignments_results_dir = "./matrices/{}/fixed_k/".format(dataset_name)
        else:
            assignments_results_dir = "./matrices/{}/random_k/".format(dataset_name)

        matrices = dict()
        matrices['BA'] = pd.read_csv(os.path.join(assignments_results_dir, 'BA.csv')).to_numpy()
        matrices['CO'] = pd.read_csv(os.path.join(assignments_results_dir, 'CO.csv')).to_numpy()
        matrices['TMB'] = pd.read_csv(os.path.join(assignments_results_dir, 'TMB.csv')).to_numpy()
        matrices['FCM'] = pd.read_csv(os.path.join(assignments_results_dir, 'FCM.csv')).to_numpy()
        matrices['WCT'] = pd.read_csv(os.path.join(assignments_results_dir, 'WCT.csv')).to_numpy()
        matrices['WTQ'] = pd.read_csv(os.path.join(assignments_results_dir, 'WTQ.csv')).to_numpy()

        return matrices

    def save_results(self, assignations, dataset_name, matrix_name, are_clusters_fixed, consensus_algorithm):
        if are_clusters_fixed:
            assignments_results_dir = "./results/{}/fixed_k/{}/".format(dataset_name, consensus_algorithm)
        else:
            assignments_results_dir = "./results/{}/random_k/{}/".format(dataset_name, consensus_algorithm)

        if not os.path.exists(assignments_results_dir):
            os.makedirs(assignments_results_dir)

        np.save(assignments_results_dir + matrix_name, assignations)

    def __get_dataset_file_path(self, dataset_name):
        root_dir = os.path.dirname(__file__)
        datasets_dir = os.path.join(root_dir, './resources/datasets')
        return os.path.join(datasets_dir, dataset_name + '.arff')
