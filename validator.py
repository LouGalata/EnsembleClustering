from sklearn.decomposition import PCA
import numpy as np
import os
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    jaccard_score,
    normalized_mutual_info_score
)


class Validator:

    def __init__(self, name, dataset, ground_truth):
        self.name = name
        self.dataset = dataset
        self.ground_truth = ground_truth
        self.labels = [a.decode('UTF8') for a in set(ground_truth)]
        self.freq_ground_truth = self.__map_values_by_frequency(self.ground_truth)
        self.pca = PCA(n_components=2)
        # Compute the PCA-transformation for the input data
        self.pc = self.pca.fit_transform(self.dataset.to_numpy())

    def internal_index_db(self, assignations):
        davies_bouldin = dict()
        for key, value in assignations.items():
            davies_bouldin[key] = davies_bouldin_score(self.dataset.to_numpy(), value)
        return davies_bouldin

    def __validate_jaccardi(self, assignations):
        jaccard_results = dict()
        for key, value in assignations.items():
            jaccard_results[key] = jaccard_score(self.freq_ground_truth, value, average='macro')
        return jaccard_results

    def __validate_nmi(self, assignations):
        nmi_results = dict()
        for key, value in assignations.items():
            nmi_results[key] = normalized_mutual_info_score(self.freq_ground_truth, value)
        return nmi_results

    def validata_single_run(self):
        assignations = dict()
        assignations['average_linkage'] = np.load(os.path.join('clusters', self.name, 'AL.npy'))
        assignations['complete_linkage'] = np.load(os.path.join('clusters', self.name, 'CL.npy'))
        assignations['single_linkage'] = np.load(os.path.join('clusters', self.name, 'SL.npy'))
        assignations['kmeans'] = np.load(os.path.join('clusters', self.name, 'Kmeans.npy'))
        assignations['kmedoids'] = np.load(os.path.join('clusters', self.name, 'Kmedoids.npy'))
        assignations['spectral'] = np.load(os.path.join('clusters', self.name, 'Spectral.npy'))
        for key, value in assignations.items():
            assignations[key] = self.__get_label_rotations(self.__map_values_by_frequency(value))

        jaccard_results = self.__validate_jaccardi(assignations)
        nmi_score = self.__validate_nmi(assignations)
        davies_bouldin = self.internal_index_db(assignations)
        self.plot_single_run_results(jaccard_results, nmi_score, davies_bouldin)
        for key, value in assignations.items():
            self.visualize(value, key)


    def plot_single_run_results(self, jaccard_results, nmi_score, davies_bouldin):
        x = list(jaccard_results.keys())
        y = list(jaccard_results.values())
        sns.set_context(rc={"figure.figsize": (8, 4)})
        plt.ylim(max(0, min(y) - 0.05), max(y) + 0.05, 1)
        plt.xticks(list(range(6)))
        fig = plt.bar(list(range(6)), y, width=len(x) / 8, color=sns.color_palette("Blues", len(x)), align='center')
        plt.legend(fig, list(x), loc="upper left", title='Algorithms', framealpha=0.2)
        plt.title(self.name + ': Jaccard Distance')
        result_path = os.path.join('results', self.name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        plt.savefig(os.path.join(result_path, 'single_run_jaccard.png'))

        x = list(nmi_score.keys())
        y = list(nmi_score.values())
        sns.set_context(rc={"figure.figsize": (8, 4)})
        plt.ylim(max(0, min(y) - 0.05), max(y) + 0.05, 1)
        plt.xticks(list(range(6)))
        fig = plt.bar(list(range(6)), y, width=len(x) / 8, color=sns.color_palette("Blues", len(x)), align='center')
        plt.legend(fig, list(x), loc="upper left", title='Algorithms', framealpha=0.2)
        plt.title(self.name + ': Normalized Mutual Info')
        plt.savefig(os.path.join(result_path, 'single_run_nmi.png'))

        x = list(davies_bouldin.keys())
        y = list(davies_bouldin.values())
        sns.set_context(rc={"figure.figsize": (8, 4)})
        plt.ylim(max(0, min(y) - 0.05), max(y) + 0.05, 1)
        plt.xticks(list(range(6)))
        fig = plt.bar(list(range(6)), y, width=len(x) / 8, color=sns.color_palette("Blues", len(x)), align='center')
        plt.legend(fig, list(x), loc="upper left", title='Algorithms', framealpha=0.2)
        plt.title(self.name + ': Davies Bouldin')
        plt.savefig(os.path.join(result_path, 'single_run_bouldin.png'))


    def __map_values_by_frequency(self, values):
        result = np.empty(np.array(values).shape)
        mapping = {}
        i = 0
        for value, _ in Counter(values).most_common():
            mapping[value] = i
            i += 1
        for i, value in enumerate(values):
            result[i] = mapping[value]
        return result


    def __get_label_rotations(self, clustering):
        n = len(clustering)
        labels = list(set(clustering))
        l = len(labels)
        rotations = [clustering]
        for i in range(1, l):
            new_values = labels[i:] + labels[:i]
            mapping = {}
            new_rotation = []
            for j in range(l):
                mapping[labels[j]] = new_values[j]
            for k in range(n):
                current_value = clustering[k]
                mapped_value = mapping[current_value]
                new_rotation.append(mapped_value)
            rotations.append(new_rotation)
        accuracy = -1
        optimal_rotation = None
        for rotation in rotations:
            new_accuracy = accuracy_score(self.freq_ground_truth, rotation)
            if new_accuracy > accuracy:
                optimal_rotation = rotation
                accuracy = new_accuracy
        return optimal_rotation


    def visualize(self, clustering, title):
        # Visualize the clustering partition using the PCA-transformation
        components = pd.DataFrame(self.pc)
        fig, ax = plt.subplots()
        colormap = np.array(self.get_random_color(len(self.labels)))
        ax.scatter(
            components[0],
            components[1],
            alpha=.4,
            c=colormap[np.array(clustering).astype(int)]
        )
        plt.xlabel('Eigenvektor 1')
        plt.ylabel('Eigenvektor 2')
        plt.title(self.name + ': ' + title)
        figure_name = "clustering_{}".format(title)
        path = os.path.join('results', self.name, figure_name)
        plt.savefig(path)


    def get_random_color(self, c):
        colors = []
        for _ in range(len(self.labels)):
            color = '#' + ''.join([str(k) for k in random.choices(list(range(0, 9)) + ['a', 'b', 'c', 'd', 'e', 'f'], k=4)])  +'ff'
            colors.append(color)
        return colors