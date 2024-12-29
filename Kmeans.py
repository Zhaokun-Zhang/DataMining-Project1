import numpy as np

class Kmeans:
    def __init__(self, n_clusters, max_iter=100, random_state=114514):
        self.max_iter = max_iter
        self.seed = random_state
        self.n_clusters = n_clusters
        self.centers = None

    def fit_predict(self, X):
        np.random.seed(42)
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([X[labels == k].mean(axis=0) 
                                    for k in range(self.n_clusters)])
            
            if np.all(self.centers == new_centers): break
            self.centers = new_centers
        
        return labels
        


