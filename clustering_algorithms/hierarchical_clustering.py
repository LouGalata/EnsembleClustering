from sklearn.cluster import AgglomerativeClustering
from enum import Enum


class Agglomerative:

    def __init__(self, data, c, ahc_linkage=None):
        self.data = data
        self.linkage = ahc_linkage
        if self.linkage is None:
            self.linkage = self.Linkage.ward
        if c is None:
            self.ahc = AgglomerativeClustering(linkage=self.linkage)
        else:
            self.ahc = AgglomerativeClustering(
                n_clusters=c,
                linkage=self.linkage
            )

    def model_name(self):
        title = "AHC (l={})"
        lin = self.linkage
        return title.format(lin)

    def clusterize(self):
        print(" * Clustering data with {}...".format(self.model_name()))
        return self.ahc.fit_predict(self.data)

    def get_model(self):
        return self.ahc

    class Linkage(Enum):
        single = 1,
        complete = 2,
        average = 3,
        ward = 4
