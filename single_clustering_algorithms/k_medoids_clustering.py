from sklearn_extra.cluster import KMedoids

class Kmedoid:

    def __init__(self, data, c=None):
        self.data = data
        self.clusters = c
        self.kmed = KMedoids(n_clusters=self.clusters)


    def model_name(self):
        title = "K Medoids"
        return title

    def clusterize(self):
        print(" * Clustering data with {}...".format(self.model_name()))
        return self.kmed.fit_predict(self.data)

    def get_model(self):
        return self.kmed

