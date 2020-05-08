import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    jaccard_score,
    normalized_mutual_info_score
)


class Validator:

    def __init__(self, name, dataset, ground_truth, consensus, are_clusters_fixed):
        self.name = name
        self.dataset = dataset
        self.ground_truth = ground_truth
        self.are_clusters_fixed = 'fixed_k' if are_clusters_fixed else 'random_k'
        self.labels = [a.decode('UTF8') for a in set(ground_truth) if isinstance(a, str)]
        self.freq_ground_truth = self.__map_values_by_frequency(self.ground_truth)
        self.pca = PCA(n_components=2)
        # Compute the PCA-transformation for the input data
        self.pc = self.pca.fit_transform(self.dataset.to_numpy())
        self.consensus = consensus

    def __internal_index_db(self, assignations):
        davies_bouldin = dict()
        for key, value in assignations.items():
            davies_bouldin[key] = davies_bouldin_score(self.dataset.to_numpy(), value)
        return davies_bouldin

    def __accuracy_score(self, assignations):
        accuracies = dict()
        for key, value in assignations.items():
            accuracies[key] = accuracy_score(self.freq_ground_truth, value)
        return accuracies

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

    def validate_consensus(self):
        assignations = dict()
        assignations['BA'] = np.load(
            os.path.join('results', self.name, self.are_clusters_fixed, self.consensus, 'BA.npy'))
        assignations['CO'] = np.load(
            os.path.join('results', self.name, self.are_clusters_fixed, self.consensus, 'CO.npy'))
        assignations['TMB'] = np.load(
            os.path.join('results', self.name, self.are_clusters_fixed, self.consensus, 'TMB.npy'))
        assignations['FCM'] = np.load(
            os.path.join('results', self.name, self.are_clusters_fixed, self.consensus, 'FCM.npy'))
        assignations['WCT'] = np.load(
            os.path.join('results', self.name, self.are_clusters_fixed, self.consensus, 'WCT.npy'))
        assignations['WTQ'] = np.load(
            os.path.join('results', self.name, self.are_clusters_fixed, self.consensus, 'WTQ.npy'))

        for key, value in assignations.items():
            assignations[key] = self.__get_label_rotations(self.__map_values_by_frequency(value))

        jaccard_results = self.__validate_jaccardi(assignations)
        nmi_score = self.__validate_nmi(assignations)
        davies_bouldin = self.__internal_index_db(assignations)
        accuracy = self.__accuracy_score(assignations)
        self.plot_consensus_results(jaccard_results, nmi_score, davies_bouldin, accuracy)
        self.save_scores(jaccard_results, nmi_score, davies_bouldin, accuracy)
        for key, value in assignations.items():
            self.visualize(value, key)

    def save_scores(self, jaccard_results, nmi_score, davies_bouldin, accuracy):
        col = ['matrix', 'jaccard', 'nmi', 'davies_bouldin', 'accuracy']
        df = pd.DataFrame(columns=col)
        df['jaccard'] = jaccard_results.values()
        df['nmi'] = nmi_score.values()
        df['davies_bouldin'] = davies_bouldin.values()
        df['accuracy'] = accuracy.values()
        df['matrix'] = list(jaccard_results.keys())
        result_path = os.path.join('out', self.name, self.are_clusters_fixed)
        df.to_csv(result_path + '/{}_scores.csv'.format(self.consensus), sep='\t', index=False, header=True)

    def plot_consensus_results(self, jaccard_results, nmi_score, davies_bouldin, accuracy):
        x = list(jaccard_results.keys())
        y = list(jaccard_results.values())
        bars = len(x)
        sns.set_context(rc={"figure.figsize": (8, 4)})
        plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
        plt.xticks(list(range(bars)))
        fig = plt.bar(list(range(bars)), y, width=len(x) / 8, color=sns.color_palette("Blues", len(x)), align='center')
        plt.legend(fig, list(x), loc="upper left", title='Matrices', framealpha=0.2)
        plt.title(self.name + ' - ' + self.consensus + ': Jaccard Distance')
        result_path = os.path.join('out', self.name, self.are_clusters_fixed)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        plt.savefig(os.path.join(result_path, self.consensus + '_jaccard_score.png'))

        x = list(nmi_score.keys())
        y = list(nmi_score.values())
        sns.set_context(rc={"figure.figsize": (8, 4)})
        plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
        plt.xticks(list(range(bars)))
        fig = plt.bar(list(range(bars)), y, width=len(x) / 8, color=sns.color_palette("Blues", len(x)), align='center')
        plt.legend(fig, list(x), loc="upper left", title='Matrices', framealpha=0.2)
        plt.title(self.name + ' - ' + self.consensus + ': Normalized Mutual Info')
        plt.savefig(os.path.join(result_path, self.consensus + '_nmi_score.png'))

        x = list(davies_bouldin.keys())
        y = list(davies_bouldin.values())
        sns.set_context(rc={"figure.figsize": (8, 4)})
        plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
        plt.xticks(list(range(bars)))
        fig = plt.bar(list(range(bars)), y, width=len(x) / 8, color=sns.color_palette("Blues", len(x)), align='center')
        plt.legend(fig, list(x), loc="upper left", title='Matrices', framealpha=0.2)
        plt.title(self.name + ' - ' + self.consensus + ': Davies Bouldin')
        plt.savefig(os.path.join(result_path, self.consensus + '_bouldin_score.png'))

        x = list(accuracy.keys())
        y = list(accuracy.values())
        sns.set_context(rc={"figure.figsize": (8, 4)})
        plt.ylim(np.min(y) - 0.05, np.max(y) + 0.05)
        plt.xticks(list(range(bars)))
        fig = plt.bar(list(range(bars)), y, width=len(x) / 8, color=sns.color_palette("Blues", len(x)), align='center')
        plt.legend(fig, list(x), loc="upper left", title='Matrices', framealpha=0.2)
        plt.title(self.name + ' - ' + self.consensus + ': Accuracy')
        plt.savefig(os.path.join(result_path, self.consensus + '_accuracy_score.png'))

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
        colormap = np.array(self.get_random_color(clustering))
        ax.scatter(
            components[0],
            components[1],
            alpha=.4,
            c=colormap[np.array(clustering).astype(int)]
        )
        plt.xlabel('Eigenvektor 1')
        plt.ylabel('Eigenvektor 2')
        plt.title(self.name + ' - ' + self.consensus + ': ' + title)
        figure_name = "scatter_plot_{}".format(title)
        path = os.path.join('out', self.name, self.are_clusters_fixed, self.consensus)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/' + figure_name)
        plt.close()

    def get_random_color(self, clusters):
        colors = []
        length = len(np.array(list(set(clusters))).astype(int))
        for _ in range(length):
            random_color = random.choices(list(range(0, 9)) + ['a', 'b', 'c', 'd', 'e', 'f'], k=4)
            color = '#' + ''.join([str(k) for k in random_color]) + 'aa'
            colors.append(color)
        return colors
