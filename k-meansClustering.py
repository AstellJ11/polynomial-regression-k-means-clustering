import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read in the data and shuffle
def get_data():
    data = pd.read_csv('Task2 - dataset - dog_breeds.csv').values

    np.random.shuffle(data)

    return data


# Find the euclidean distance between two vectors in the array
def compute_euclidean_distance(vec_1, vec_2):
    distance = ((vec_1 - vec_2) ** 2)  # Distance between two points
    distance = np.sqrt(np.sum(distance))

    return distance


# Create random centroid values within the size of the dataset
def initialise_centroids(dataset, k):
    centroids = dataset[np.random.randint(dataset.shape[0], size=k)]

    return centroids[:k]


# def kmeans(dataset, k):
#    centroids = initialise_centroids(dataset, k)  # Centroids are created at random
#
#    return centroids, cluster_assigned


data = get_data()

plt.scatter(data[:, 0], data[:, 1], )
plt.xlabel("Height")
plt.ylabel("Tail Length")
plt.savefig('k-meanClustering.png')
plt.show()

test_centroids = initialise_centroids(data, 2)
print(test_centroids)
