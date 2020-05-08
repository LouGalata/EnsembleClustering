from file_manager import FileManager
import os
import numpy as np
from preprocessing import Preprocessor
from clustering_algorithms.hierarchical_clustering import Agglomerative
from clustering_algorithms.k_means_clustering import KMeansAlgorithm
from clustering_algorithms.k_medoids_clustering import Kmedoid
from clustering_algorithms.spectral_clustering import Spectral
from clustering_algorithms.metis_clustering import MetisClustering
from validator import Validator
from clustering_algorithms.fuzzy_c_means import FuzzyMeansAlgorithm
from matrix_constructor import MatrixConstructor
from consensus_strategies import Consensus
import math
import random
import scipy
import networkx as nx


class Experiment:
    ENSEMBLING_EXPERIMENTS = 10

    HIERARCHICAL_STRATEGIES = ['AL', 'SL', 'CL']

    def __init__(self, real_or_artificial, dataset, real_number_of_classes, are_clusters_fixed):
        self.name = dataset
        self.real_number_of_classes = real_number_of_classes
        dataset_path = os.path.join(real_or_artificial, dataset)
        self.are_clusters_fixed = are_clusters_fixed
        self.dataset, self.meta = FileManager().load_arff(dataset_path)
        self.num_rows = len(self.dataset.index)
        self.num_random_c = random.choice(range(2, int(math.sqrt(self.num_rows)))) if math.sqrt(self.num_rows) < 50 else random.choice(range(2, 50))
        self.num_fixed_c = int(math.sqrt(self.num_rows)) if math.sqrt(self.num_rows) < 50 else 50
        self.validator = None
        self.ground_truth = None
        self.numerical = None
        self.algorithm = None


    def run_single_clusterings(self):
        self.preprocess()

        is_fixed = 'fixed_k' if self.are_clusters_fixed else 'random_k'
        if len(os.listdir(os.path.join('clusters', self.name, is_fixed, 'Kmeans'))) != 10:
            for i in range(Experiment.ENSEMBLING_EXPERIMENTS):
                print("Single crispy clustering: \033[1m{}\033[0m (N={}) exp={}\n".format(self.name, self.num_rows, i))
                FileManager().save_csv(self.numerical, self.name)
                _, gt_indices = np.unique(self.ground_truth, return_inverse=True)
                c = self.num_fixed_c if self.are_clusters_fixed else self.num_random_c
                self.__train_kmeans_ensemblers(c, i)
                print()

        if len(os.listdir(os.path.join('clusters', self.name, is_fixed, 'FuzzyMeans'))) != 10:
            for i in range(Experiment.ENSEMBLING_EXPERIMENTS):
                print("Single fuzzy clustering: \033[1m{}\033[0m (N={}) exp={}\n".format(self.name, self.num_rows, i))
                FileManager().save_csv(self.numerical, self.name)
                _, gt_indices = np.unique(self.ground_truth, return_inverse=True)
                c = self.num_fixed_c if self.are_clusters_fixed else self.num_random_c
                self.__train_fuzzy_c_means_ensemblers(c, i)
                print()

    def preprocess(self):
        p = Preprocessor(self.dataset)
        p.preprocess()
        self.ground_truth = p.get_classification_data().values
        self.numerical = p.get_numerical()

# ENSEMBLING METHODS
    def __train_fuzzy_c_means_ensemblers(self, c, exp_num):
        numerical = self.numerical.to_numpy()
        model = FuzzyMeansAlgorithm(numerical, c)
        assignations = model.clusterize()
        FileManager().save_assignments(assignations.fuzzy_labels_, self.name, exp_num, self.are_clusters_fixed, algorithm='FuzzyMeans')

    def __train_kmeans_ensemblers(self, c, exp_num):
        numerical = self.numerical.to_numpy()
        assignations = []
        cl = self.num_fixed_c if self.are_clusters_fixed else self.num_random_c
        while (len(set(assignations)) < cl):
            model = KMeansAlgorithm(numerical, c)
            assignations = model.clusterize()
        FileManager().save_assignments(assignations, self.name, exp_num, self.are_clusters_fixed, algorithm='Kmeans')


# CONSENSUS METHODS
    def __train_ahc_model(self, c, link=None, algorithm='SL'):
        matrices = FileManager().get_ensembling_matrices(self.name, self.are_clusters_fixed)
        for key, value in matrices.items():
            model = Agglomerative(value, c, ahc_linkage=link)
            assignations = model.clusterize()
            FileManager().save_results(assignations, self.name, key, self.are_clusters_fixed, algorithm)

        Validator(self.name, self.numerical, self.ground_truth, algorithm, self.are_clusters_fixed).validate_consensus()


    def __train_kmedoids_model(self, c):
        matrices = FileManager().get_ensembling_matrices(self.name, self.are_clusters_fixed)
        for key, value in matrices.items():
            model = Kmedoid(value, c)
            assignations = model.clusterize()
            FileManager().save_results(assignations.labels_, self.name, key, self.are_clusters_fixed, 'Kmedoids')
        Validator(self.name, self.numerical, self.ground_truth, 'Kmedoids', self.are_clusters_fixed).validate_consensus()


    def __train_kmeans_model(self, c):
        matrices = FileManager().get_ensembling_matrices(self.name, self.are_clusters_fixed)
        for key, value in matrices.items():
            model = KMeansAlgorithm(value, c)
            assignations = model.clusterize()
            FileManager().save_results(assignations, self.name, key, self.are_clusters_fixed, 'Kmeans')
        Validator(self.name, self.numerical, self.ground_truth, 'Kmeans', self.are_clusters_fixed).validate_consensus()


    def __train_spectral_model(self, c):
        matrices = FileManager().get_ensembling_matrices(self.name, self.are_clusters_fixed)
        for key, value in matrices.items():
            model = Spectral(value, c)
            assignations = model.clusterize()
            FileManager().save_results(assignations, self.name, key, self.are_clusters_fixed, 'Spectral')
        Validator(self.name, self.numerical, self.ground_truth, 'Spectral', self.are_clusters_fixed).validate_consensus()


    def __train_metis_model(self, c):
        matrices = FileManager().get_ensembling_matrices(self.name, self.are_clusters_fixed)
        for key, value in matrices.items():
            if key == 'CO':
                model = MetisClustering(value, c)
                assignations = model.clusterize()[1]
                FileManager().save_results(assignations, self.name, key, self.are_clusters_fixed, 'Metis')
            else:
                features = len(value[0])
                graph = nx.Graph()
                graph.add_nodes_from(list(range(self.num_rows)))
                graph.add_nodes_from(list(range(self.num_rows, self.num_rows+features)))
                for cnt_row, rows in enumerate(value):
                    for cnt_col, col in enumerate(rows):
                        graph.add_edge(cnt_row, self.num_rows + cnt_col, weight=col)

                model = MetisClustering(value, c, graph)
                assignations = model.clusterize()[1][:self.num_rows]
                FileManager().save_results(assignations, self.name, key, self.are_clusters_fixed, 'Metis')
        Validator(self.name, self.numerical, self.ground_truth, 'Metis',
                      self.are_clusters_fixed).validate_consensus()

    def create_ensembling_matrices(self):
        print("Ensemble clustering: \033[1m{}\033[0m (N={})\n".format(self.name, self.num_rows))
        create_ensemble_matrices = MatrixConstructor(
            name=self.name,
            are_clusters_fixed=self.are_clusters_fixed,
            num_random_c=self.num_random_c,
            num_fixed_c=self.num_fixed_c)
        create_ensemble_matrices.load_matrices(Experiment.ENSEMBLING_EXPERIMENTS)

    def run_ensemling_clusterings(self, consensus_strategy=None):
        if consensus_strategy in Experiment.HIERARCHICAL_STRATEGIES:
            model = Consensus.__members__.get(consensus_strategy).value
            linkage_type = model.split(' ')[0].lower()
            self.__train_ahc_model(self.real_number_of_classes,linkage_type, consensus_strategy)

        elif consensus_strategy == 'KM':
            self.__train_kmeans_model(self.real_number_of_classes)

        elif consensus_strategy == 'PAM':
            self.__train_kmedoids_model(self.real_number_of_classes)

        elif consensus_strategy == 'SPC':
            self.__train_spectral_model(self.real_number_of_classes)

        elif consensus_strategy == 'METIS':
            self.__train_metis_model(self.real_number_of_classes)

        else:
            print('Method {} is either not implemented or wrong written'.format(consensus_strategy))
            print('Avaiable strategies are: %s' %(Consensus.__members__.keys()))



