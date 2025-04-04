#imports
from kmeans_algorithm import KMEANSClustering
from hac_algorithm import HACClustering
from load_data import load_iris_data, load_phonome_data

# Load the datasets
iris_input_features = load_iris_data()
phonome_input_features = load_phonome_data()

# CHANGE K-VALUES HERE
iris_k_kmeans = 3
phonome_k_kmeans = 5
iris_k_HAC = 3
phonome_k_HAC = 5

# CHANGE CUSTOM CENTROIDS FOR KMEANS HERE
custom_centroids = [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3],
    [5.8, 2.7, 4.1, 1.0]
]

def run_kmeans():
    print("\n====== K-MEANS RESULTS ======\n")
    
    # Run K-Means on the iris dataset
    kmeans_iris_data = KMEANSClustering(iris_k_kmeans)
    kmeans_iris_data.fit(iris_input_features)
    print(f"-----K-Means Clustering on the Iris Dataset (k={iris_k_kmeans})-----")
    kmeans_iris_data.print_labels()
    print()
    
    # Run K-Means on the phonome dataset
    kmeans_phonome_data = KMEANSClustering(phonome_k_kmeans)
    kmeans_phonome_data.fit(phonome_input_features)
    print(f"-----K-Means Clustering on the Phonome Dataset (k={phonome_k_kmeans})-----")
    kmeans_phonome_data.print_labels()
    print()
    
    custom_k = len(custom_centroids)
    kmeans_iris_custom_centroids = KMEANSClustering(custom_k, initial_centroids=custom_centroids)
    kmeans_iris_custom_centroids.fit(iris_input_features)
    print(f"-----K-Means Clustering on Iris with Specifying k initial centroids-----")
    kmeans_iris_custom_centroids.print_labels()
    print("Custom Centroids for reference:")
    print(custom_centroids)
    print(f"(k={custom_k})")

def run_hac():
    print("\n====== HAC RESULTS ======\n")
    
    # Run HAC on the iris dataset
    hac_iris_single = HACClustering(iris_k_HAC, link_type='single')
    hac_iris_single.fit(iris_input_features)
    print(f"-----HAC Clustering on the Iris Dataset (k={iris_k_HAC}, Single Link)-----")
    hac_iris_single.print_labels()
    print()
    
    hac_iris_complete = HACClustering(iris_k_HAC, link_type='complete')
    hac_iris_complete.fit(iris_input_features)
    print(f"-----HAC Clustering on the Iris Dataset (k={iris_k_HAC}, Complete Link)-----")
    hac_iris_complete.print_labels()
    print()
    
    print("Note: HAC on Phonome dataset is very slow and commented out by default")
    # # Uncomment these to run HAC on the phonome dataset
    # hac_phonome_single = HACClustering(phonome_k_HAC, link_type='single')
    # hac_phonome_single.fit(phonome_input_features)
    # print(f"-----HAC Clustering on the Phonome Dataset (k={phonome_k_HAC}, Single Link)-----")
    # hac_phonome_single.print_labels()
    # print()
    
    # hac_phonome_complete = HACClustering(phonome_k_HAC, link_type='complete')
    # hac_phonome_complete.fit(phonome_input_features)
    # print(f"-----HAC Clustering on the Phonome Dataset (k={phonome_k_HAC}, Complete Link)-----")
    # hac_phonome_complete.print_labels()

if __name__ == "__main__":
    print("Running clustering algorithms...")
    
    # Run K-Means
    run_kmeans()
    
    # Run HAC
    run_hac()
    
    print("\nAll clustering algorithms completed.") 