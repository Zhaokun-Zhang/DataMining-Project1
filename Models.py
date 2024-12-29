import numpy as np
from Kmeans import Kmeans
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

class Model:
    def __init__(self, model_name, params, seed=114514):
        self.model_name = model_name
        self.n_clusters = params['n_clusters']
        self.centers = None
        if model_name[0] == 'k': # k-means
            self.model = Kmeans(**params, random_state=seed)
        elif model_name[0] == 'a': # AgglomerativeClustering
            self.model = AgglomerativeClustering(**params)
        elif model_name[0] == 'm': # MiniBatch Kmeans
            self.model = MiniBatchKMeans(**params, random_state=seed)

    def fit_predict(self, X):
        labels = self.model.fit_predict(X)
        # calculate centers
        if self.model_name[0] == 'k': # k-means
            self.centers = self.model.centers
        elif self.model_name[0] == 'a': # AgglomerativeClustering
            self.centers = np.array([X[labels==k].mean(axis=0) for k in range(self.n_clusters)])
        return labels
            


    def predict(self, X):
        if self.model_name[0] == 'm': # MiniBatch Kmeans
            return self.model.predict(X)
        else:
            return self.center_predict(X)
            

    def center_predict(self, X):
        return np.argmin(np.linalg.norm(X[:,np.newaxis] - self.centers, axis=2), axis=1)