from sklearn.cluster import SpectralClustering


class Spectral:

    def __init__(self, data, c):
        self.data = data
        self.n_clusters = c
        if c is None: self.spc = SpectralClustering(n_jobs=2)
        else: self.spc = SpectralClustering(n_clusters=self.n_clusters, n_jobs=2)

    def model_name(self):
        title = "Spectral Algorithm"
        return title

    def clusterize(self):
        print(" * Clustering data with {}...".format(self.model_name()))
        return self.spc.fit_predict(self.data)

    def get_model(self):
        return self.spc