from sklearn_extra.cluster import KMedoids
from sklearn_extensions.fuzzy_kmeans import KMedians


class Kmedoid:

    def __init__(self, data, c=None):
        self.data = data
        self.clusters = c
        if c is None: self.kmed = KMedians()
        else: self.kmed = KMedians(self.clusters)


    def model_name(self):
        title = "K Medoids"
        return title

    def clusterize(self):
        print(" * Clustering data with {}...".format(self.model_name()))
        return self.kmed.fit(self.data)

    def get_model(self):
        return self.kmed

