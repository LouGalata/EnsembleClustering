from file_manager import FileManager
import os
import numpy as np
from preprocessing import Preprocessor
from single_clustering_algorithms.hierarchical_clustering import Agglomerative
from single_clustering_algorithms.k_means_clustering import KMeansAlgorithm
from single_clustering_algorithms.k_medoids_clustering import Kmedoid
from single_clustering_algorithms.spectral_clustering import Spectral
from validator import Validator


class Experiment():

    def __init__(self, real_or_artificial, dataset, real_number_of_classes):
        self.name = dataset
        self.clusters = real_number_of_classes
        dataset_path = os.path.join(real_or_artificial, dataset)
        self.dataset, self.meta = FileManager().load_arff(dataset_path)
        self.validator = None
        self.ground_truth = None
        self.numerical = None

    def run_single_clusterings(self):
        num_rows = len(self.dataset.index)
        print("Single clustering: \033[1m{}\033[0m (N={})\n".format(self.name, num_rows))
        self.preprocess()
        FileManager().save_csv(self.numerical, self.name)
        _, gt_indices = np.unique(self.ground_truth, return_inverse=True)
        print()
        self.__exp_iteration(random_c=False)


    def preprocess(self):
        p = Preprocessor(self.dataset)
        p.preprocess()
        self.ground_truth = p.get_classification_data().values
        self.numerical = p.get_numerical()


    def __exp_iteration(self, random_c=False):
        if not random_c:
            c = self.clusters
        self.__train_ahc_model(c, link=Agglomerative.Linkage.single, algorithm='SL')
        self.__train_ahc_model(c, link=Agglomerative.Linkage.average, algorithm='AL')
        self.__train_ahc_model(c, link=Agglomerative.Linkage.complete, algorithm='CL')

        self.__train_kmedoids_model(c)
        self.__train_kmeans_model(c)
        self.__train_spectral_model(c)

        Validator(self.name, self.numerical, self.ground_truth).validata_single_run()
        pass

    def __train_ahc_model(self, c, link=None, algorithm='SL'):
        numerical = self.numerical.to_numpy()
        model = Agglomerative(numerical, c, ahc_linkage=link)
        assignations = model.clusterize()
        FileManager().save_assignments(assignations, self.name, algorithm)

    def __train_kmedoids_model(self, c):
        numerical = self.numerical.to_numpy()
        model = Kmedoid(numerical, c)
        assignations = model.clusterize()
        FileManager().save_assignments(assignations, self.name, algorithm='Kmedoids')

    def __train_kmeans_model(self, c):
        numerical = self.numerical.to_numpy()
        model = KMeansAlgorithm(numerical, c)
        assignations = model.clusterize()
        FileManager().save_assignments(assignations, self.name, algorithm='Kmeans')
        pass

    def __train_spectral_model(self, c):
        numerical = self.numerical.to_numpy()
        model = Spectral(numerical, c)
        assignations = model.clusterize()
        FileManager().save_assignments(assignations, self.name, algorithm='Spectral')
        pass

    def run_ensemling_clusterings(self):
        num_rows = len(self.dataset.index)
        print("Ensemble clustering: \033[1m{}\033[0m (N={})\n".format(self.name, num_rows))
        pass
