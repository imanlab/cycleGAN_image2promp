"""
This file is used to to normalize the weights data and save them in a json file.
"""
import os
import numpy as np
import json
import scipy.stats as stats

from matplotlib import pyplot as plt
from sklearn import preprocessing

dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca/"
path_to_json = dataset_dir + "probabilistic_7dof/"
# take the json files in path_to_json
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith(".json")]
NUM_FILES = len(json_files)  # num of json_files in path_to_json
NUM_CONFIG = 5
NUM_STRAWBERRIES = 20
normalization_range = (0, 255)


class Scaling:
    """
    I need this class to save the scaler for each strawberry so that later I can de-normalize the right way
    """
    def __init__(self, strawberry, to_norm_mean, to_norm_eig, to_norm_cov, norm_mean, norm_eig, norm_cov,
                 scaler_mean, scaler_eig, scaler_cov):
        self.strawberry = strawberry
        self.to_norm_mean = to_norm_mean
        self.to_norm_eig = to_norm_eig
        self.to_norm_cov = to_norm_cov
        self.norm_mean = norm_mean
        self.norm_eig = norm_eig
        self.norm_cov = norm_cov
        self.scaler_mean = scaler_mean
        self.scaler_eig = scaler_eig
        self.scaler_cov = scaler_cov


def empty_array():
    """
    :return: the function returns an empty float32 numpy array to be filled
    """
    list = []

    return np.array(list, dtype='float32')


if __name__ == "__main__":

    """
    THE NEXT PART IS TO CREATE THE JSON FILES WITH THE NORMALIZED WEIGHTS
    """
    scaler_list = []
    to_norm_mean = empty_array()
    to_norm_eig = empty_array()
    to_norm_cov = empty_array()
    # Load the to be normalized weights_dir (from 'probabilistic_7dof')
    for i in range(NUM_CONFIG):
        for j in range(NUM_STRAWBERRIES):
            skip = False
            if j < 10:
                try:
                    with open(path_to_json + str(i) + "0" + str(j) + "_ConfigStrawberry"
                                                                     "_mean&covVectors.json") as f:
                        data_w = json.load(f)
                        # name of the strawberry for the saving in the class
                        strawberry_name = str(i) + "0" + str(j) + "_ConfigStrawberry"
                except FileNotFoundError:
                    print("File " + str(i) + "0" + str(j) + "_ConfigStrawberry_mean&covVectors.json ==> "
                                                            "missing.. skipping to the next file")
                    skip = True
            else:
                try:
                    with open(path_to_json + str(i) + str(j) + "_ConfigStrawberry"
                                                               "_mean&covVectors.json") as f:
                        data_w = json.load(f)
                        # name of the strawberry for the saving in the class
                        strawberry_name = str(i) + str(j) + "_ConfigStrawberry"
                except FileNotFoundError:
                    print("File " + str(i) + str(j) + "_ConfigStrawberry_mean&covVectors.json ==> "
                                                      "missing.. skipping to the next file")
                    skip = True
            f.close()
            if not skip:
                '''define the scaler for the normalization'''
                minmax_mean_scaler = preprocessing.MinMaxScaler(feature_range=normalization_range)
                minmax_eig_scaler = preprocessing.MinMaxScaler(feature_range=normalization_range)
                minmax_cov_scaler = preprocessing.MinMaxScaler(feature_range=normalization_range)

                for k in range(7):
                    w_mean = data_w[k]['mean_vector']
                    w_eig = data_w[k]['covariance_vector'][0]
                    w_cov = data_w[k]['covariance_vector'][1:]
                    mean_vector = np.array(w_mean, dtype='float32')
                    eig_vector = np.array(w_eig, dtype='float32')
                    cov_vector = np.array(w_cov, dtype='float32')

                    # UNCOMMENT NEXT 5 LINES FOR CREATING THE WEIGHT IMAGE
                    # if k == 0:
                    #     mean_cov_vector = np.concatenate((mean_vector, cov_vector))
                    # else:
                    #     fake = np.concatenate((mean_vector, cov_vector))  # first concatenation of mean and cov
                    #     mean_cov_vector = np.concatenate((mean_cov_vector, fake))  # total concatenation

                    # UNCOMMENT THE NEXT 4 LINES FOR THE NORMALIZATION PART
                    to_norm_mean = np.concatenate((to_norm_mean, mean_vector))
                    eig_vector = np.expand_dims(eig_vector, 0)  # needed to concatenate
                    to_norm_eig = np.concatenate((to_norm_eig, eig_vector))
                    to_norm_cov = np.concatenate((to_norm_cov, cov_vector))

                """
                NEXT LINES FOR THE NORMALIZATION WRT EACH STRAWBERRY WITH THE MINMAX
                """
                to_norm_mean = np.reshape(to_norm_mean, (-1, 1))  # reshape to perform scaling
                to_norm_eig = np.reshape(to_norm_eig, (-1, 1))
                to_norm_cov = np.reshape(to_norm_cov, (-1, 1))

                minmax_mean = minmax_mean_scaler.fit_transform(to_norm_mean)
                minmax_eig = minmax_eig_scaler.fit_transform(to_norm_eig)
                minmax_cov = minmax_cov_scaler.fit_transform(to_norm_cov)
                # original_mean = minmax_mean_scaler.inverse_transform(minmax_mean) ==> to go back to the original values

                '''squeeze a dimension'''
                minmax_mean = np.squeeze(minmax_mean)
                minmax_eig = np.squeeze(minmax_eig)
                minmax_cov = np.squeeze(minmax_cov)

                scaler = Scaling(strawberry_name, to_norm_mean, to_norm_eig, to_norm_cov,
                                 minmax_mean, minmax_eig, minmax_cov,
                                 minmax_mean_scaler, minmax_eig_scaler, minmax_cov_scaler)
                scaler_list.append(scaler)
                # Let's reconstruct the normalized json file
                mean_eig_cov_vector = []
                for k in range(7):
                    mean = minmax_mean[(k * 8):((k + 1) * 8)]
                    eig = minmax_eig[k]
                    cov = minmax_cov[(k * 8):((k + 1) * 8)]
                    eig = np.expand_dims(eig, 0)
                    eig_cov = np.concatenate((eig, cov))
                    jsonlist = {"mean_vector": mean.tolist(), "covariance_vector": eig_cov.tolist()}
                    mean_eig_cov_vector.append(jsonlist)
                    mean = empty_array()
                    eig = empty_array()
                    cov = empty_array()

                filepath = dataset_dir + "probabilistic_normalized_255/" + strawberry_name + \
                           "_0To255Normalized_mean&covVectors.json"
                with open(filepath, 'w') as f:
                    json.dump(mean_eig_cov_vector, f)

                to_norm_mean = empty_array()
                to_norm_eig = empty_array()
                to_norm_cov = empty_array()


    """
    THE NEXT PART IS FOR THE DE-NORMALIZATION 
    
    to_validate = np.load("/home/francesco/PycharmProjects/cycleGAN_image_to_weight/training_15_01_epoch130/test_5.npy")

    # Bring the shape to a 256x256, computing the mean across the RGB values
    to_validate = np.squeeze(to_validate)   # predicted.shape is (1, 256, 256, 3) => remove dim 1
    to_validate_r = to_validate[:, :, 0]    # take the R value
    to_validate_g = to_validate[:, :, 1]    # take the G value
    to_validate_b = to_validate[:, :, 2]    # take the B value
    to_validate_meaned = (to_validate_r + to_validate_g + to_validate_b) / 3     # compute the mean between the RGB values
    to_validate_meaned = to_validate_meaned[:, :238]
    # Let's reconstruct the 119 element vector
    to_validate_vector = []
    val = 0
    for k in range(238):
        for i in range(256):
            val = val + to_validate_meaned[i, k]
        if (((k + 1) % 2) == 0):
            val = val / 512
            to_validate_vector.append(val)
            val = 0
    to_validate_vector = np.asarray(to_validate_vector, dtype="float32")
    # Separate mean, eig, cov to de-normalize
    full_mean = empty_array()
    full_eig = empty_array()
    full_cov = empty_array()
    to_val_mean_eig_cov = []
    for k in range(7):
        dof = to_validate_vector[(k * 17):((k + 1) * 17)]
        to_val_mean = dof[0:8]
        to_val_eig = dof[8]
        to_val_eig = np.expand_dims(to_val_eig, 0)  # needed to concatenate
        to_val_cov = dof[9:]
        to_val_eig_cov = np.concatenate((to_val_eig, to_val_cov))
        to_val_list = {"mean_vector": to_val_mean.tolist(), "covariance_vector": to_val_eig_cov.tolist()}   # to create the json
        to_val_mean_eig_cov.append(to_val_list)
        full_mean = np.concatenate((full_mean, to_val_mean))
        full_eig = np.concatenate((full_eig, to_val_eig))
        full_cov = np.concatenate((full_cov, to_val_cov))

    with open("/home/francesco/prova_to_denorm_test5.json", 'w') as p:
        json.dump(to_val_mean_eig_cov, p)

    full_mean = full_mean.reshape(-1, 1)
    full_eig = full_eig.reshape(-1, 1)
    full_cov = full_cov.reshape(-1, 1)

    # In the next operations, 0 is associated to the strawberry image => need to implement
    # a loop to match the strawberry autonomously
    denormalized_mean = scaler_list[84].scaler_mean.inverse_transform(full_mean)
    denormalized_eig = scaler_list[84].scaler_eig.inverse_transform(full_eig)
    denormalized_cov = scaler_list[84].scaler_cov.inverse_transform(full_cov)

    denormalized_mean = np.squeeze(denormalized_mean)
    denormalized_eig = np.squeeze(denormalized_eig)
    denormalized_cov = np.squeeze(denormalized_cov)

    denorm_mean_eig_cov = []
    for k in range(7):
        denorm_mean = denormalized_mean[(k * 8):((k + 1) * 8)]
        denorm_eig = denormalized_eig[k]
        denorm_cov = denormalized_cov[(k * 8):((k + 1) * 8)]
        denorm_eig = np.expand_dims(denorm_eig, 0)
        denorm_eig_cov = np.concatenate((denorm_eig, denorm_cov))
        denormalized_list = {"mean_vector": denorm_mean.tolist(), "covariance_vector": denorm_eig_cov.tolist()}
        denorm_mean_eig_cov.append(denormalized_list)
        denorm_mean = empty_array()
        denorm_eig = empty_array()
        denorm_cov = empty_array()

    denorm_path = "/home/francesco/PycharmProjects/cycleGAN_image_to_weight/training_15_01_epoch130/denorm_test_5.json"
    with open(denorm_path, 'w') as f:
        json.dump(denorm_mean_eig_cov, f)
    """


    """
    NEXT LINES FOR NORMALIZATION SUBTRACTING THE AVERAGE VALUE TO EACH ELEMENT 

    mean_mean = np.mean(np.abs(to_norm_mean))
    mean_eig = np.mean(to_norm_eig)
    mean_cov = np.mean(np.abs(to_norm_cov))

    ''' Subtract (or add) the mean value to each element '''
    for i in range(len(to_norm_mean)):
        if (to_norm_mean[i] > 0):
            to_norm_mean[i] = to_norm_mean[i] - mean_mean
        else:
            to_norm_mean[i] = to_norm_mean[i] + mean_mean

    adjusted_eig = to_norm_eig - mean_eig

    for i in range(len(to_norm_cov)):
        if (to_norm_cov[i] > 0):
            to_norm_cov[i] = to_norm_cov[i] - mean_cov
        else:
            to_norm_cov[i] = to_norm_cov[i] + mean_cov


    to_norm_mean = np.reshape(to_norm_mean, (-1, 1))  # reshape to perform scaling
    to_norm_eig = np.reshape(to_norm_eig, (-1, 1))
    to_norm_cov = np.reshape(to_norm_cov, (-1, 1))

    minmax_mean_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    minmax_eig_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    minmax_cov_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    minmax_mean = minmax_mean_scaler.fit_transform(to_norm_mean)
    minmax_eig = minmax_eig_scaler.fit_transform(to_norm_eig)
    minmax_cov = minmax_cov_scaler.fit_transform(to_norm_cov)
    # original_mean = minmax_mean_scaler.inverse_transform(minmax_mean) ==> to go back to the original values

    '''squeeze a dimension'''
    minmax_mean = np.squeeze(minmax_mean)
    minmax_eig = np.squeeze(minmax_eig)
    minmax_cov = np.squeeze(minmax_cov)

    # NEXT LINES TO SAVE THE NORMALIZED JSON FILE, IN CASE len(minmax_mean) = 5320

    mean = empty_array()
    eig = empty_array()
    cov = empty_array()
    mean_cov_vector = []
    cont = 0
    i = 0
    k = 0
    j = 0
    while i < len(minmax_mean):
        skip = False
        cont = cont + 1
        mean = np.append(mean, minmax_mean[i])
        cov = np.append(cov, minmax_cov[i])
        if ((cont % 8) == 0):
            cov = np.append(minmax_eig[int((cont / 8) - 1)], cov)
            jsonlist = {"mean_vector": mean.tolist(), "covariance_vector": cov.tolist()}
            mean_cov_vector.append(jsonlist)
            mean = empty_array()
            eig = empty_array()
            cov = empty_array()
            if ((cont % 56) == 0):
                if ((str(j) + str(k) == "019") or (str(j) + str(k) == "115") or (str(j) + str(k) == "118") or (
                        str(j) + str(k) == "218") or (str(j) + "0" + str(k) == "302")):
                    skip = True
                    i = i - 56
                    cont = cont - 56
                if not skip:
                    if k < 10:
                        filepath = dataset_dir + "/dataset_pca/probabilistic_normalized_byaverage/" + str(j) + "0" + str(k) \
                                   + "_ConfigStrawberry_NormalizedByAverage_mean&covVectors.json"
                        with open(filepath, 'w') as f:
                            json.dump(mean_cov_vector, f)
                    else:
                        filepath = dataset_dir + "/dataset_pca/probabilistic_normalized_byaverage/" + str(j) + str(k) \
                                   + "_ConfigStrawberry_NormalizedByAverage_mean&covVectors.json"
                        with open(filepath, 'w') as f:
                            json.dump(mean_cov_vector, f)
                if k == 19:
                    k = 0
                    j = j + 1
                else:
                    k = k + 1
                mean_cov_vector = []
        i = i + 1
    """

    """
    norm_eig = np.linalg.norm(to_norm_eig)
    norm_mean = np.linalg.norm(to_norm_mean)    # for 2D arrays, the standard norm is the Frobenius norm
    norm_cov = np.linalg.norm(to_norm_cov)      # for 2D arrays, the standard norm is the Frobenius norm
    eig_normalized = to_norm_eig / norm_eig
    mean_normalized = to_norm_mean / norm_mean
    cov_normalized = to_norm_cov / norm_cov

    standard_eig = preprocessing.StandardScaler().fit_transform(to_norm_eig)
    standard_minmax_mean = minmax.fit_transform(standard_mean)
    minmax_standard_mean = standard.fit_transform(minmax_mean)


    # THE NEXT LINES OF CODE ARE JUST TO PLOT THE HISTROGRAMS OF THE NORMALIZED DISTRIBUTION 

    plt.figure()

    plt.subplot(1, 3, 1)
    q25, q75 = np.percentile(to_norm_mean, [25, 75])
    bin_width = 2 * (q75 - q25) * len(to_norm_mean) ** (-1 / 3)
    bins1 = round((to_norm_mean.max() - to_norm_mean.min()) / bin_width)  # Freedman-Diaconis rule for bins
    plt.hist(to_norm_mean, density=True, bins=bins1)
    plt.title("to_norm_mean")
    plt.ylabel("Probability")
    plt.xlabel("Data")

    plt.subplot(1, 3, 2)
    q25, q75 = np.percentile(to_norm_eig, [25, 75])
    bin_width = 2 * (q75 - q25) * len(to_norm_eig) ** (-1 / 3)
    bins2 = round((to_norm_eig.max() - to_norm_eig.min()) / bin_width)
    plt.hist(to_norm_eig, density=True, bins=bins2)
    plt.title("to_norm_eig")
    plt.ylabel("Probability")
    plt.xlabel("Data")

    plt.subplot(1, 3, 3)
    q25, q75 = np.percentile(to_norm_cov, [25, 75])
    bin_width = 2 * (q75 - q25) * len(to_norm_cov) ** (-1 / 3)
    bins3 = round((to_norm_cov.max() - to_norm_cov.min()) / bin_width)
    plt.hist(to_norm_cov, density=True, bins=bins3)
    plt.title("to_norm_cov")
    plt.ylabel("Probability")
    plt.xlabel("Data")

    plt.show()

    plt.figure()
    q25, q75 = np.percentile(standard_eig, [25, 75])
    bin_width = 2 * (q75 - q25) * len(standard_eig) ** (-1 / 3)
    bins = round((standard_eig.max() - standard_eig.min()) / bin_width)
    plt.hist(standard_eig, density=True, bins=bins)
    plt.title("standard_eig")
    plt.ylabel("Probability")
    plt.xlabel("Data")
    plt.show()


    plt.figure()

    plt.subplot(2, 2, 1)
    q25, q75 = np.percentile(standard_mean, [25, 75])
    bin_width = 2 * (q75 - q25) * len(standard_mean) ** (-1 / 3)
    bins1 = round((standard_mean.max() - standard_mean.min()) / bin_width)  # Freedman-Diaconis rule for bins 
    plt.hist(standard_mean, density=True, bins=bins1)
    plt.title("standard_mean")
    plt.ylabel("Probability")
    plt.xlabel("Data")

    plt.subplot(2, 2, 2)
    q25, q75 = np.percentile(minmax_mean, [25, 75])
    bin_width = 2 * (q75 - q25) * len(minmax_mean) ** (-1 / 3)
    bins2 = round((minmax_mean.max() - minmax_mean.min()) / bin_width)
    plt.hist(minmax_mean, density=True, bins=bins2)
    plt.title("minmax_mean")
    plt.ylabel("Probability")
    plt.xlabel("Data")

    plt.subplot(2, 2, 3)
    q25, q75 = np.percentile(standard_minmax_mean, [25, 75])
    bin_width = 2 * (q75 - q25) * len(standard_minmax_mean) ** (-1 / 3)
    bins3 = round((standard_minmax_mean.max() - standard_minmax_mean.min()) / bin_width)
    plt.hist(standard_minmax_mean, density=True, bins=bins3)
    plt.title("standard_minmax_mean")
    plt.ylabel("Probability")
    plt.xlabel("Data")

    plt.subplot(2, 2, 4)
    q25, q75 = np.percentile(minmax_standard_mean, [25, 75])
    bin_width = 2 * (q75 - q25) * len(minmax_standard_mean) ** (-1 / 3)
    bins4 = round((minmax_standard_mean.max() - minmax_standard_mean.min()) / bin_width)
    plt.hist(minmax_standard_mean, density=True, bins=bins4)
    plt.title("minmax_standard_mean")
    plt.ylabel("Probability")
    plt.xlabel("Data")

    plt.show()
    """