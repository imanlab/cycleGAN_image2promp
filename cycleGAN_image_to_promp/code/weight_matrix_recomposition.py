"""
This file is used to reconstruct the covariance matrix, after the PCA was applied.
"""
import numpy as np
import json


def reconstruct_covariance(covariance_weights):
    s = covariance_weights[0]   # eigenvalue
    u = covariance_weights[1:]  # covariance values
    uu = u.reshape((int(len(u)), 1))
    covariance_matrix = np.dot(np.dot(uu, s), uu.T)

    return covariance_matrix


if __name__ == "__main__":

    # Test the function:
    dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca"
    with open(dataset_dir + "/probabilistic_7dof/000_ConfigStrawberry_mean&covVectors.json", 'r') as fp:
        annotation = json.load(fp)
        mean_weights = np.asarray(annotation[0]['mean_vector']).round(16).astype('float64')
        covariance_weights = np.asarray(annotation[0]['covariance_vector'])
        # the number in square brackets is the dof we are looking at
    covariance_matrix = reconstruct_covariance(covariance_weights)
    fp.close()
