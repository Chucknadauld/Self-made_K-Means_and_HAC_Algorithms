from scipy.io import arff
import pandas as pd

"""LOAD IN THE EXAMPLE DATASETS"""
# Iris dataset
def load_iris_data():
    iris_data = arff.loadarff("example_datasets/iris_data.arff")
    iris_df = pd.DataFrame(iris_data[0])
    return iris_df.drop(columns=["class"])

# Phonome dataset
def load_phonome_data():
    phoneme_data = arff.loadarff("example_datasets/phonome_data.arff")
    phoneme_df = pd.DataFrame(phoneme_data[0])
    return phoneme_df.drop(columns=["Class"])
