import networkx as nx
import numpy as np
import metis
import random

class MetisClustering:

    def __init__(self, data, c, graph=None):
        if not graph:
            self.data = nx.from_numpy_matrix(np.array(data))
            self.data.graph['edge_weight_attr'] = 'weight'
        else:
            self.data = graph
        self.n_clusters = c
        if c is None: self.n_clusters = random.choice(2, 10)


    def model_name(self):
        title = "Metis"
        return title

    def clusterize(self):
        print(" * Clustering data with {}...".format(self.model_name()))
        return metis.part_graph(self.data, nparts=self.n_clusters, recursive=True)

