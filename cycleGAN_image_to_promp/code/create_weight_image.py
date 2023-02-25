"""
This file was used to create the different types of weight images used for the trainings.
"""
import os
import numpy as np
import json

from matplotlib import pyplot as plt
from sklearn import preprocessing
from image_average import noisy

dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca/"
path_to_json = dataset_dir + "probabilistic_normalized_255/"
# take the json files in path_to_json
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith(".json")]
NUM_FILES = len(json_files)  # num of json_files in path_to_json
NUM_CONFIG = 5
NUM_STRAWBERRIES = 20


def pixel_data_mean(data):

    pixel = np.zeros((3, 3, 3), dtype="float32")  # (23, 23, 3) for the 'weight_image_square'
    pixel[:, :] = (data, 0, 0)

    return pixel


def pixel_data_eig(data):

    pixel = np.zeros((3, 3, 3), dtype="float32")  # (23, 23, 3) for the 'weight_image_square'
    pixel[:, :] = (0, data, 0)

    return  pixel


def pixel_data_cov(data):

    pixel = np.zeros((3, 3, 3), dtype="float32")  # (23, 23, 3) for the 'weight_image_square'
    pixel[:, :] = (0, 0, data)

    return pixel


def pixel_data_general(data):
    """
    :param data: weight value from the json file
    :return: 23x23x3 square with 'data' value on each pixel
    """
    pixel = np.zeros((23, 23, 3), dtype="float32")
    pixel[:, :] = (data, data, data)

    return pixel


def weight_square_33(data, average):
    """
    :param data: weight values from the json file
    :param average: average values from the json file
    :return: weight_square 33x33x3 by pixel. R == mean, G == eig, B == cov.
    """
    pixel_list = []
    for k in range(7):
        w_mean = (np.array(data[k]['mean_vector'], dtype="float32")) - (np.array(average[k]['mean_vector'], dtype="float32"))
        w_cov = (np.array(data[k]['covariance_vector'], dtype="float32")) - (np.array(average[k]['covariance_vector'], dtype="float32"))
        w_mean = w_mean.round().astype("uint8")
        w_cov = w_cov.round().astype("uint8")
        eig_vector = w_cov[0]
        cov_vector = w_cov[1:]
        for i in range(len(w_mean)):
            pixel_list.append(pixel_data_mean(w_mean[i]))
        pixel_list.append(pixel_data_eig(eig_vector))
        for i in range(len(cov_vector)):
            pixel_list.append(pixel_data_cov(cov_vector[i]))

    # Add two 3x3x3 zeros matrix, to reach 11x11 "3x3 pixels" in both rows and columns
    pixel_padding = np.zeros((3, 3, 3), dtype="uint8")
    pixel_list.append(pixel_padding)
    pixel_list.append(pixel_padding)
    pixel_rows = []
    for i in range(len(pixel_list)):
        if (i == 0 or (i % 11) == 0):  # or i % 11 to know when to go to the next row
            row = pixel_list[i]
        else:
            row = np.concatenate((row, pixel_list[i]), axis=1)
        if (((i + 1) % 11) == 0):
            pixel_rows.append(row)

    for j in range(11):
        if j == 0:
            weight_im = pixel_rows[0]
        else:
            weight_im = np.concatenate((weight_im, pixel_rows[j]), axis=0)

    return weight_im


def weight_image_square(data):
    """
    :param data: numpy float32 weight vector of size 1x119
    :return: weight_image 256x256x3 by pixel. R == mean, G == eig, B == cov
    """
    pixel_list = []
    for k in range(7):
        w_mean = data[k]['mean_vector']
        w_cov = data[k]['covariance_vector']
        mean_vector = np.array(w_mean, dtype='float32')
        eig_vector = np.array(w_cov[0], dtype="float32")
        cov_vector = np.array(w_cov[1:], dtype='float32')
        for i in range(len(mean_vector)):
            pixel_list.append(pixel_data_mean(mean_vector[i]))
        pixel_list.append(pixel_data_eig(eig_vector))
        for i in range(len(cov_vector)):
            pixel_list.append(pixel_data_cov(cov_vector[i]))

    # Add two 23x23x3 zeros matrix, to reach 11x11 "big pixels" in both rows and columns
    pixel_padding = np.zeros((23, 23, 3), dtype="float32")
    pixel_list.append(pixel_padding)
    pixel_list.append(pixel_padding)
    pixel_rows = []
    for i in range(len(pixel_list)):
        if (i == 0 or (i % 11) == 0):   # or i % 11 to know when to go to the next row
            row = pixel_list[i]
        else:
            row = np.concatenate((row, pixel_list[i]), axis=1)
        if (((i + 1) % 11) == 0):
            pixel_rows.append(row)

    for j in range(11):
        if j == 0:
            weight_im = pixel_rows[0]
        else:
            weight_im = np.concatenate((weight_im, pixel_rows[j]), axis=0)
    row_padding = np.zeros([256, 3, 3], dtype="float32")
    col_padding = np.zeros([3, 253, 3], dtype="float32")
    weight_im = np.concatenate((weight_im, col_padding), axis=0)  # add 3 columns
    weight_im = np.concatenate((weight_im, row_padding), axis=1)  # add 3 rows

    return weight_im


def weight_image_rowcol(data):
    """
    :param data: numpy float32 weight vector of size 1x119
    :return: weight image 256x256x3 stacked in rows and columns
    """
    N = 18  # number of 0 value pixels to be added
    data_row = np.concatenate((data, data))
    data_row = np.pad(data_row, (0, N), 'constant')  # stack N 0s to the right end of the vector
    weight_im = np.tile(data_row, [256, 1])  # stacking on rows in order to get a 256x256 shape
    weight_im = np.expand_dims(weight_im, axis=2)
    weight_im = np.repeat(weight_im, 3, axis=2)

    return weight_im


def weight_image_pixel(data):
    """
    :param data: numpy float32 weight vector of size 1x119
    :return: weight image 256x256x3 stacked in pixels
    """
    N = 2  # number of 0 value pixels to be added to each vector
    data = np.pad(data, (0, N), 'constant')
    data_pixel = np.reshape(data, [11, 11])  # reshape the weight into a pixel-like
    weight_im = np.tile(data_pixel, [23, 23])  # stacking to obtain 253x253 shape
    # we need to add some zeros in order to fulfill 256x256 shape
    weight_im = np.pad(weight_im, (0, 3), 'constant')
    weight_im = np.expand_dims(weight_im, axis=2)
    weight_im = np.repeat(weight_im, 3, axis=2)

    return weight_im


def weight_image_splitted(data):
    """
    :param data: numpy float32 weight vector of size 1x119
    :return: weight image 256x256x3 with just one value every two columns
    """
    N = 18  # number of 0 values columns to be padded
    for i in range(len(data)):
        single_weight = data[i]
        single_weight_tiled = np.tile(single_weight, [256, 2])
        if i == 0:
            weight_im = single_weight_tiled
        else:
            weight_im = np.append(weight_im, single_weight_tiled, axis=1)
    weight_im = np.pad(weight_im, ((0, 0), (0, N)), 'constant')  # stack N 0s to the right end of the matrix
    weight_im = np.expand_dims(weight_im, axis=2)
    weight_im = np.repeat(weight_im, 3, axis=2)

    return weight_im


def empty_array():
    """
    :return: the function returns an empty float32 numpy array to be filled
    """
    list = []

    return np.array(list, dtype='float32')


if __name__ == "__main__":

    """
    NEXT LINES ARE FOR THE CREATION OF THE NORMALIZED WEIGHT IMAGES
    """
    first = True
    config_list = []
    # Import the average vector from the json file
    with open(dataset_dir + "json_average_255.json") as av:
        average = json.load(av)
    av.close()
    # Load the weights_dir
    for i in range(NUM_CONFIG):
        for j in range(NUM_STRAWBERRIES):
            skip = False
            if j < 10:
                try:
                    with open(path_to_json + str(i) + "0" + str(j) + "_ConfigStrawberry_0To255Normalized"
                                                                     "_mean&covVectors.json") as f:
                        data_w = json.load(f)
                except FileNotFoundError:
                    print("File " + str(i) + "0" + str(j) + "_ConfigStrawberry_SingleBerryNormalized_mean&covVectors"
                                                            ".json ==> missing.. skipping to the next file")
                    skip = True
            else:
                try:
                    with open(path_to_json + str(i) + str(j) + "_ConfigStrawberry_0To255Normalized"
                                                               "_mean&covVectors.json") as f:
                        data_w = json.load(f)
                except FileNotFoundError:
                    print("File " + str(i) + str(j) + "_ConfigStrawberry_SingleBerryNormalized_mean&covVectors.json "
                                                      "==> missing.. skipping to the next file")
                    skip = True
            f.close()
            if not skip:
                image_square = weight_square_33(data_w, average)

                # We now want to create a image with noise, and put 'image_square' in the middle
                white = np.full((256, 256, 3), 255, dtype="uint8")
                white[112:145, 112:145, :] = image_square

                # for k in range(7):
                #     w_mean = data_w[k]['mean_vector']
                #     w_cov = data_w[k]['covariance_vector']
                #     mean_vector = np.array(w_mean, dtype='float32')
                #     cov_vector = np.array(w_cov, dtype='float32')
                #     if k == 0:
                #         mean_cov_vector = np.concatenate((mean_vector, cov_vector))
                #     else:
                #         fake = np.concatenate((mean_vector, cov_vector))              # first concatenation of mean and cov
                #         mean_cov_vector = np.concatenate((mean_cov_vector, fake))     # total concatenation

                """
                TO COMPUTE THE AVERAGE VECTOR 
                
                if first:
                    average = mean_cov_vector / 95
                    first = False
                else:
                    average = average + mean_cov_vector / 95
                """
                # image_rowcol = weight_image_rowcol(mean_cov_vector)
                # image_pixel = weight_image_pixel(mean_cov_vector)
                # image_splitted = weight_image_splitted(mean_cov_vector)
                if j < 10:
                    np.save("/home/francesco/PycharmProjects/dataset/dataset_pca/normalized_weight_images_255_sub_nonoise/"
                            + str(i) + "0" + str(j) + "_weight_image_255_sub_nonoise", white)
                else:
                    np.save("/home/francesco/PycharmProjects/dataset/dataset_pca/normalized_weight_images_255_sub_nonoise/"
                            + str(i) + str(j) + "_weight_image_255_sub_nonoise", white)
