# Self-made K-Means and HAC Algorithms

This repository contains implementations of K-means and Hierarchical Agglomerative Clustering (HAC) algorithms from scratch. These are fundamental unsupervised learning algorithms used for grouping data points based on similarity.

## Project Structure

- **kmeans_algorithm.py**: Implementation of the K-means clustering algorithm
  - Customizable number of clusters (k)
  - Support for custom initial centroids
  - Debug mode for deterministic initialization

- **hac_algorithm.py**: Implementation of the Hierarchical Agglomerative Clustering algorithm
  - Support for both single and complete linkage methods
  - Customizable number of clusters (k)

- **load_data.py**: Utility functions to load and preprocess datasets
  - Handles ARFF file format
  - Prepares data for clustering algorithms

- **run_algorithms.py**: Runner script to execute algorithms and display results
  - Configurable k values for each algorithm
  - Option to specify custom centroids
  - HAC on Phonome dataset is commented out by default (very slow)

## Datasets

### Iris Dataset
- **Creator**: R.A. Fisher
- **Donor**: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
- **Date**: July, 1988
- **Citation**: Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936)
- **Description**: Contains 3 classes of 50 instances each, where each class refers to a type of iris plant.

### Phonome Dataset
- **Author**: Dominique Van Cappel, THOMSON-SINTRA
- **Source**: KEEL, ELENA - 1993
- **Origin**: European ESPRIT 5516 project: ROARS
- **Description**: Dataset to distinguish between nasal (class 0) and oral sounds (class 1). Five attributes (amplitudes of the first five harmonics normalized by total energy) characterize each vowel. Contains 5404 instances from 1809 isolated syllables.

## Usage

1. Ensure you have the required dependencies:
   ```
   uv pip install scikit-learn scipy pandas
   ```

2. Run the clustering algorithms:
   ```
   python run_algorithms.py
   ```

3. To modify parameters (like k values or custom centroids), edit the configuration section at the top of `run_algorithms.py`.

## Notes

- The HAC algorithm with the Phonome dataset is very computationally intensive (O(n²) complexity) and may take a long time to run.
- The K-means implementation includes a convergence check to avoid unnecessary iterations.
- Both algorithms inherit from scikit-learn's BaseEstimator and ClusterMixin for compatibility with the scikit-learn ecosystem.