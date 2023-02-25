"""
This file is to show the selected images.
Mostly used to check the newly created weight images.
"""
import numpy as np
from matplotlib import pyplot as plt
import json

dataset_dir = "/home/francesco/PycharmProjects/dataset/"
path_to_json = dataset_dir + "dataset_pca/normalized_weight_images_square/"
# take the json files in path_to_json
NUM_CONFIG = 5
NUM_STRAWBERRIES = 20


if __name__ == "__main__":

    for i in range(NUM_CONFIG):
        for j in range(NUM_STRAWBERRIES):
            skip = False
            if j < 10:
                try:
                    data = np.load(path_to_json + str(i) + "0" + str(j) + "_weight_image_square.npy")
                except FileNotFoundError:
                    print("File " + str(i) + "0" + str(j) + "_ConfigStrawberry_SingleBerryNormalized_mean&covVectors"
                                                            ".json ==> missing.. skipping to the next file")
                    skip = True
            else:
                try:
                    data = np.load(path_to_json + str(i) + str(j) + "_weight_image_255.npy")
                except FileNotFoundError:
                    print("File " + str(i) + str(j) + "_ConfigStrawberry_SingleBerryNormalized_mean&covVectors.json "
                                                      "==> missing.. skipping to the next file")
                    skip = True
            if not skip:
                plt.figure()
                plt.imshow(data)
                plt.show()
