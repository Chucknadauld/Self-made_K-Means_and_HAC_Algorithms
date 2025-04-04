#imports
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from load_data import load_iris_data, load_phonome_data

# Load the datasets
iris_input_features = load_iris_data()
phonome_input_features = load_phonome_data()

class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,link_type='single'):
        self.link_type = link_type
        self.k = k
        self.labels = None
        
    def fit(self, X, y=None):
        X = np.array(X)
        n_clusters = X.shape[0]

        # Make each point its own cluster
        cluster = [[i] for i in range(n_clusters)]
        active_clusters = list(range(n_clusters))

        # Calculate the pairwise distance of all the points
        distance = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                distance_formula = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distance[i, j] = distance_formula
                distance[j, i] = distance_formula

        # Keep merging until we have k clusters
        current_k = n_clusters
        while current_k > self.k:
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i_idx, i in enumerate(active_clusters):
                for j_idx, j in enumerate(active_clusters[i_idx+1:]):
                    j_idx = i_idx + 1 + j_idx
                    
                    # Find the distance between each cluster according to linkage type
                    if self.link_type == 'single':
                        cluster_dist = float('inf')
                        for point_i in cluster[i]:
                            for point_j in cluster[j]:
                                if distance[point_i, point_j] < cluster_dist:
                                    cluster_dist = distance[point_i, point_j]
                    else:
                        # For complete link
                        cluster_dist = 0
                        for point_i in cluster[i]:
                            for point_j in cluster[j]:
                                if distance[point_i, point_j] > cluster_dist:
                                    cluster_dist = distance[point_i, point_j]
                    
                    if cluster_dist < min_dist:
                        min_dist = cluster_dist
                        merge_i, merge_j = i, j
            
            cluster[merge_i].extend(cluster[merge_j])
            active_clusters.remove(merge_j)  # Mark as inactive
            current_k -= 1
        
        # Give the cluster labels
        self.labels = np.zeros(n_clusters, dtype=int)
        for cluster_idx, c in enumerate(active_clusters):
            for point_idx in cluster[c]:
                self.labels[point_idx] = cluster_idx
                
        return self
    
    def print_labels(self): # Print the cluster label for each data point
        print("Cluster labels:\n", self.labels)

        # Added this to understand the results better
        print(f"Results:")
        unique_labels = np.unique(self.labels)
        for i in unique_labels:
            count = np.sum(self.labels == i)
            print(f"   Cluster {i}: {count} points")

# CHANGE K-VALUES HERE
iris_k_HAC = 3
phonome_k_HAC = 5

# Run HAC on the two datasets
hac_iris_single = HACClustering(iris_k_HAC, link_type='single')
hac_iris_single.fit(iris_input_features)
hac_iris_complete = HACClustering(iris_k_HAC, link_type='complete')
hac_iris_complete.fit(iris_input_features)

hac_phonome_single = HACClustering(phonome_k_HAC, link_type='single')
hac_phonome_single.fit(phoneme_input_features)
hac_phonome_complete = HACClustering(phonome_k_HAC, link_type='complete')
hac_phonome_complete.fit(phoneme_input_features)

# Print results
print(f"-----HAC Clustering on the Iris Dataset (k={iris_k_HAC}, Single Link)-----")
hac_iris_single.print_labels()
print()
print(f"-----HAC Clustering on the Iris Dataset (k={iris_k_HAC}, Complete Link)-----")
hac_iris_complete.print_labels()
print()

print(f"-----HAC Clustering on the Phonome Dataset (k={phonome_k_HAC}, Single Link)-----")
hac_phonome_single.print_labels()
print()
print(f"-----HAC Clustering on the Phonome Dataset (k={phonome_k_HAC}, Complete Link)-----")
hac_phonome_complete.print_labels()