import os

import networkx as nx
import numpy as np

from file_manager import FileManager


class MatrixConstructor:
    DECAY = 0.9

    def __init__(self, name=None, are_clusters_fixed=None, num_random_c=None, num_fixed_c=None):
        self.name = name
        self.num_rows = None
        self.are_clusters_fixed = 'fixed_k' if are_clusters_fixed else 'random_k'
        self.num_random_c = num_random_c
        self.num_fixed_c = num_fixed_c
        self.total_clusters = 0
        self.BA = None
        self.ra_graph = None
        self.ra_num_clusters_of_each_partition = None
        self.triples_labels = None

    def load_matrices(self, num_ensemblers):
        base_clusterings = []
        fuzzy_base_clusterings = []
        for i in range(num_ensemblers):
            base_clusterings.append(np.load(
                os.path.join('clusters', self.name, self.are_clusters_fixed, 'Kmeans', 'ensembling_{}.npy'.format(i))))

        for i in range(num_ensemblers):
            fuzzy_base_clusterings.append(np.load(
                os.path.join('clusters', self.name, self.are_clusters_fixed, 'FuzzyMeans',
                             'ensembling_{}.npy'.format(i))))

        self.num_random_c = len(set(base_clusterings[0]))
        if self.are_clusters_fixed == 'fixed_k':
            self.total_clusters = self.num_fixed_c * len(base_clusterings)
        else:
            self.total_clusters = self.num_random_c * len(base_clusterings)
        self.num_rows = len(base_clusterings[0])

        BA_matrix = self.__build_binary_association_matrix(base_clusterings)
        CO_matrix = self.__build_coassociation_matrix(base_clusterings)
        TMB_matrix = self.__build_TMB_matrix()
        # TODO: FCM needs soft clustering
        FCM_matrix = self.__build_FCM_matrix(fuzzy_base_clusterings)
        # WDM_matrix = self.__build_WDM_matrix()
        RA_WCT_matrix = self.__build_RA_WCT_matrix(base_clusterings)
        RA_WTQ_matrix = self.__build_RA_WTQ_matrix(base_clusterings)

        # Save matrices
        FileManager().save_ensembling_matrices(self.name, BA_matrix, 'BA', self.are_clusters_fixed)
        FileManager().save_ensembling_matrices(self.name, CO_matrix, 'CO', self.are_clusters_fixed)
        FileManager().save_ensembling_matrices(self.name, TMB_matrix, 'TMB', self.are_clusters_fixed)
        FileManager().save_ensembling_matrices(self.name, FCM_matrix, 'FCM', self.are_clusters_fixed)
        FileManager().save_ensembling_matrices(self.name, RA_WCT_matrix, 'WCT', self.are_clusters_fixed)
        FileManager().save_ensembling_matrices(self.name, RA_WTQ_matrix, 'WTQ', self.are_clusters_fixed)

    def __build_pairwise_similarity_matrix(self, base_clustering):
        pairwise_matrix = np.zeros((self.num_rows, self.num_rows), dtype=np.float)
        for idx_x, x in enumerate(base_clustering):
            for idx_y, y in enumerate(base_clustering):
                if idx_y == idx_x: pairwise_matrix[idx_x, idx_y] = 1
                if idx_y > idx_x and base_clustering[idx_y] == base_clustering[idx_x]:
                    pairwise_matrix[idx_x, idx_y] = 1
                    pairwise_matrix[idx_y, idx_x] = 1
        return pairwise_matrix

    def __build_binary_association_matrix(self, base_clusterings):
        binary_clustering_matrix = np.zeros((self.num_rows, self.total_clusters), dtype=np.int)
        cnt = 0
        for base_clustering in base_clusterings:
            num_clusters = len(set(base_clustering))
            a = np.array(base_clustering).astype(np.int)
            binary_clustering_matrix[np.arange(a.size), a - 1 + cnt] = 1
            cnt += num_clusters
        self.BA = binary_clustering_matrix
        return binary_clustering_matrix

    def __build_coassociation_matrix(self, base_clusterings):
        pairwise_matrices = [self.__build_pairwise_similarity_matrix(base_clustering) for base_clustering in
                             base_clusterings]
        return np.sum(pairwise_matrices, axis=0) / self.total_clusters

    def __build_TMB_matrix(self):
        probabilities = np.sum(self.BA, axis=0) / self.num_rows
        return self.BA - probabilities

    def __build_FCM_matrix(self, fuzzy_based_clusterings):
        tmp = fuzzy_based_clusterings[0]
        for fuzzy_based_clustering in fuzzy_based_clusterings[1:]:
            tmp = np.concatenate((tmp, fuzzy_based_clustering), axis=-1)
        return tmp

    def __build_WDM_matrix(self):
        pass

    def __build_RA_WCT_matrix(self, base_clusterings):
        self.ra_graph, self.ra_num_clusters_of_each_partition = self.__create_ra_graph_helper(base_clusterings)
        self.triples_labels = self.__get_graph_triples()
        wct_node = dict()
        for key, value in self.triples_labels.items():
            n1, n2 = key
            wct_node[key] = sum(
                [min(abs(self.ra_graph[n1][a]['weight']), abs(self.ra_graph[n2][a]['weight'])) for a in value])
            wct_node[(n2, n1)] = wct_node[key]
        wct_max = max(wct_node.values())

        for key, value in wct_node.items():
            wct_node[key] = (wct_node[key] / wct_max) * MatrixConstructor.DECAY

        for node in self.ra_graph.nodes():
            wct_node[(node, node)] = 1.0

        ra_wct_clustering_matrix = np.zeros((self.num_rows, self.total_clusters), dtype=np.float)
        for i in range(self.num_rows):
            cnt = 0
            for j in range(self.total_clusters):
                if j > self.ra_num_clusters_of_each_partition[cnt] - 1: cnt += 1
                # TODO: remove the -1
                current_assignment = base_clusterings[cnt][i]
                if cnt > 0: current_assignment += self.ra_num_clusters_of_each_partition[cnt - 1]
                ra_wct_clustering_matrix[i, j] = wct_node[(current_assignment, j)]

        return ra_wct_clustering_matrix

    def __build_RA_WTQ_matrix(self, base_clusterings):
        n_nodes = self.ra_graph.number_of_nodes()
        w_centers = np.zeros(n_nodes, dtype=np.float)
        for node in range(n_nodes):
            neighbors = self.ra_graph.neighbors(node)
            w_centers[node] = 1.0 / (sum([self.ra_graph[node][n]['weight'] for n in neighbors]))

        wtq_node = dict()
        for key, value in self.triples_labels.items():
            n1, n2 = key
            wtq_node[key] = sum([w_centers[v] for v in value])
            wtq_node[(n2, n1)] = wtq_node[key]
        wtq_max = max(wtq_node.values())

        for key, value in wtq_node.items():
            wtq_node[key] = (wtq_node[key] / wtq_max) * MatrixConstructor.DECAY

        for node in self.ra_graph.nodes():
            wtq_node[(node, node)] = 1.0

        ra_wtq_clustering_matrix = np.zeros((self.num_rows, self.total_clusters), dtype=np.float)
        for i in range(self.num_rows):
            cnt = 0
            for j in range(self.total_clusters):
                if j > self.ra_num_clusters_of_each_partition[cnt] - 1: cnt += 1
                # TODO: remove the -1
                # current_assignment = base_clusterings[cnt][i] - 1
                current_assignment = base_clusterings[cnt][i]
                if cnt > 0: current_assignment += self.ra_num_clusters_of_each_partition[cnt - 1]
                ra_wtq_clustering_matrix[i, j] = wtq_node[(current_assignment, j)]

        return ra_wtq_clustering_matrix

    def __create_ra_graph_helper(self, base_clusterings):
        graph = nx.Graph()
        L = np.empty(len(base_clusterings), dtype=object)
        num_clusters_of_each_partition = np.zeros(len(base_clusterings), dtype=np.int)
        for cnt, base_clustering in enumerate(base_clusterings):
            L[cnt] = []
            if cnt > 0:
                num_clusters_of_each_partition[cnt] = num_clusters_of_each_partition[cnt - 1] + len(
                    set(base_clustering))
            else:
                num_clusters_of_each_partition[cnt] = len(set(base_clustering))
            if cnt > 0:
                num_clusters_of_partition = num_clusters_of_each_partition[cnt] - num_clusters_of_each_partition[
                    cnt - 1]
            else:
                num_clusters_of_partition = num_clusters_of_each_partition[0]
            for i in range(num_clusters_of_partition):
                # TODO: Remove the +1 after the testing
                # L[cnt].append(list(np.where(np.array(base_clustering).astype(int) == i + 1)[0]))
                L[cnt].append(list(np.where(np.array(base_clustering).astype(int) == i)[0]))

        graph.add_nodes_from(list(range(self.total_clusters)))
        for i in range(len(base_clusterings)):
            for j in range(len(base_clusterings)):
                if j > i:
                    tempA = L[i].copy()
                    tempB = L[j].copy()
                    for cnt1, t1 in enumerate(tempA):
                        for cnt2, t2 in enumerate(tempB):
                            w = len(set(t1) & set(t2)) / len(set(t1) | set(t2))
                            if i > 0:
                                nodeA = num_clusters_of_each_partition[i - 1] + cnt1
                            else:
                                nodeA = cnt1
                            nodeB = num_clusters_of_each_partition[j - 1] + cnt2
                            # print(nodeA, nodeB, w)
                            graph.add_weighted_edges_from([(nodeA, nodeB, w)])
                            graph.add_weighted_edges_from([(nodeB, nodeA, w)])

        return graph, num_clusters_of_each_partition

    def __get_graph_triples(self):
        triples_labels = dict()
        for i in self.ra_graph.nodes():
            for j in self.ra_graph.nodes():
                if j > i:
                    if not (i, j) in self.ra_graph.edges():
                        neighbors_i = [n for n in self.ra_graph.neighbors(i) if self.ra_graph[i][n]['weight'] > 0]
                        neighbors_j = [n for n in self.ra_graph.neighbors(j) if self.ra_graph[j][n]['weight'] > 0]
                        shared_neighbors = set(neighbors_i) & set(neighbors_j)
                        triples_labels[(i, j)] = shared_neighbors
        return triples_labels

    def matrices_tester(self, base_clusterings):
        self.num_rows = 5
        self.total_clusters = 5
        self.__build_RA_WCT_matrix(base_clusterings)
        self.__build_RA_WTQ_matrix(base_clusterings)
