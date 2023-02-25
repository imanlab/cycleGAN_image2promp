"""
This file is used to perform the preprocessing on the images before feeding them into the network.
"""
import tensorflow as tf
import os, os.path


BUFFER_SIZE = 100  # 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def getListOfFiles(dirName):
    """
    :param dirName: directory to the folder you want to get files from
            listOfFiles: list of each file inside the folder
    :return: allFiles: list of paths of each file inside the folder
    """
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # iterate over all the entries
    for entry in listOfFile:
        # create full path
        fullPath = os.path.join(dirName, entry)
        allFiles.append(fullPath)

    return allFiles


def resize(image, height, width):
    image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image


def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    # Next line commented because we already cast the tensor in load_image function
    # image = tf.cast(image, tf.float32)  # casts a tensor to a new type (float32) --> PROBABLY CHANGE TO FLOAT16
    image = (image / 127.5) - 1  # normalize between [-1, 1]. Default by tensorflow site

    return image


@tf.function()
def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = resize(image, 286, 286)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)
    # random mirroring
    # image = tf.image.random_flip_left_right(image)

    return image


# If I don't read the image PyCharm doesn't recognize it. The code-line 'tf.io.read_file(image_file)' is crucial
def load_image(image_file):
    # Read and decode an image file to an uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image)
    image = tf.cast(image, tf.float32)

    return image


def preprocess_image_train(image_file):
    image = load_image(image_file)
    image = random_jitter(image)
    image = resize(image, IMG_WIDTH, IMG_HEIGHT)
    image = normalize(image)

    return image


def preprocess_image_test(image_file):
    image = load_image(image_file)
    image = resize(image, IMG_WIDTH, IMG_HEIGHT)
    image = normalize(image)

    return image
