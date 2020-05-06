from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

class FuzzyMeansAlgorithm:

    def __init__(self, data, c):
        self.data = data
        self.n_clusters = c
        # Fuzzy index set to 2 in order to be aligned to the paper
        if c is None: self.fuzzy_means = FuzzyKMeans(m=2)
        else:
            self.fuzzy_means = FuzzyKMeans(self.n_clusters, m=2)

    def model_name(self):
        title = "Fuzzy-Means"
        return title

    def clusterize(self):
        print(" * Clustering data with {}...".format(self.model_name()))
        return self.fuzzy_means.fit(self.data)

    def get_model(self):
        return self.fuzzy_means