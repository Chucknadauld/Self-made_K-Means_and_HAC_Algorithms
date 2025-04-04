# imports
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,debug=False,initial_centroids=None):
        # if debug = true then use the first k instances as the initial centroids 
        # if debug = false then choose random points as the initial centroids
        # if initial_centroids is given, use those points as centroids
        self.k = k
        self.debug = debug
        self.centroids = None
        self.labels = None
        self.initial_centroids = initial_centroids

    def fit(self, X, y=None):
        X = np.array(X)
    
        if self.initial_centroids is not None:
            self.centroids = np.array(self.initial_centroids).copy()
        elif self.debug:
            self.centroids = X[:self.k].copy()
        else:
            self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)].copy()

        self.labels = np.zeros(X.shape[0], dtype=int)

        max_iterations = 50
        converge = False
        iteration = 0

        # Calculate distance to the nearest centroid
        while not converge and iteration < max_iterations:
            previous_centroids = self.centroids.copy()

            for k in range(X.shape[0]):
                distances = []
                for i in range(self.k):
                    distance = np.sqrt(np.sum((X[k] - self.centroids[i]) ** 2))
                    distances.append(distance)

                self.labels[k] = np.argmin(distances)

            # Update the centroids
            for j in range(self.k):
                cluster_points = X[self.labels == j]
                if len(cluster_points) > 0:
                    self.centroids[j] = np.mean(cluster_points, axis=0)

            # Check for convergence by seeing if centroids changed a lot
            if np.allclose(previous_centroids, self.centroids):
                converge = True
            iteration += 1

        return self
    
    def print_labels(self): # Print the cluster label for each data point
        print("Cluster labels:\n", self.labels)

        # Added this to understand the results better
        print(f"Results:")
        for i in range(self.k):
            count = np.sum(self.labels == i)
            print(f"   Cluster {i}: {count} points")