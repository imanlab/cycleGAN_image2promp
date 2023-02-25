"""
This file was used to compute the average weight image and the noise.
"""
import os
import numpy as np
import cv2
import random

from matplotlib import pyplot as plt
from PIL import Image


def noisy(image, noise_typ):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.5
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        out = np.zeros(image.size, np.uint8)
        prob = 0.05
        thres = 1 - prob
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                rdn = random.random()
                if rdn < prob:
                    out[i][j] = 0
                elif rdn > thres:
                    out[i][j] = 255
                else:
                    out[i][j] = image[i][j]
        return out
        # col, row = image.size
        # ch = 3
        # s_vs_p = 0.5
        # amount = 0.004
        # out = np.copy(image)
        # # Salt mode
        # num_salt = np.ceil(amount * image.size() * s_vs_p)
        # coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        # out[coords] = 1
        #
        # # Pepper mode
        # num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        # coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        # out[coords] = 0
        # return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


if __name__ == "__main__":

    """
    NEXT PART IS FOR COMPUTING THE AVERAGE IMAGE
    
    # Access all weight_images_coloured files in directory
    weights_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca/normalized_weight_images_255/"
    new_dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca/normalized_weight_images_255_sub/"
    wlist = [filename for filename in os.listdir(weights_dir) if filename.endswith('.npy')]
    # Assuming all images are the same size, get dimensions of first image
    w, h, c = np.load(weights_dir + wlist[0]).shape
    N = len(wlist)
    
    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, c), dtype="float32")

    # Build up average pixel intensities, casting each image as an array of floats
    for weight in wlist:
        imarr = np.array(np.load(weights_dir + weight), dtype='float32')
        imarr = noisy(imarr, "gauss")
        plt.figure()
        plt.imshow(imarr.astype("uint8"))
        plt.show()
        
        # PLOT TO SEE THE DIFFERENCE IN float32 AND uint8 IMAGES 
        a = imarr.astype("uint8")
        plt.subplot(121)
        plt.imshow(imarr)
        plt.subplot(122)
        plt.imshow(a)
        plt.show()
        
        arr = arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype='uint8')
    # Remove the average image from the single image
    for weight in wlist:
        w = np.array(np.load(weights_dir + weight).round(), dtype='uint8')
        final = w - arr
        np.save(new_dataset_dir + weight.replace("255.npy", "255_sub.npy"), final)
    """

    strawberry_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca/strawberry_whites/"
    new_dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca/strawberry_whites_noise/"
    wlist = [filename for filename in os.listdir(strawberry_dir) if filename.endswith('.png')]
    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(strawberry_dir + wlist[0]).size
    N = len(wlist)

    for weight in wlist:
        w = Image.open(strawberry_dir + weight)
        w = noisy(w, "gauss").astype("uint8")
        img = Image.fromarray(w, mode="RGB")
        # img.save(new_dataset_dir + weight)
