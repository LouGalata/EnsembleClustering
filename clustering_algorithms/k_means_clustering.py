from sklearn.cluster import KMeans


class KMeansAlgorithm:

    def __init__(self, data, c):
        self.data = data
        self.n_clusters = c
        if c is None: self.kmean = KMeans(n_jobs=2)
        else:
            self.kmean = KMeans(n_clusters=self.n_clusters, n_jobs=2)

    def model_name(self):
        title = "K-Means"
        return title

    def clusterize(self):
        print(" * Clustering data with {}...".format(self.model_name()))
        return self.kmean.fit_predict(self.data)

    def get_model(self):
        return self.kmean