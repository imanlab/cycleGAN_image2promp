"""
This file is used to train the cycleGAN model.
"""
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import os, os.path
import time
import pathlib
import imageio
import numpy as np

from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from IPython.display import clear_output
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from IPython import display
from config import *
from model import *
from preprocessing import *

""" Input Pipeline """
AUTOTUNE = tf.data.AUTOTUNE  # It is an optimization tool

BUFFER_SIZE = 100  # 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
# NUM_TRAIN_WEIGHTS = int(len(os.listdir(PATH + "weight_images_rowcol/train/")))
# NUM_TEST_WEIGHTS = int(len(os.listdir(PATH + "weight_images_rowcol/test/")))

# Load the annotation for the boundary box
json_files = sorted([pos_json for pos_json in os.listdir(PATH_TO_BB) if pos_json.endswith('.json')])
strawberry_train_files = os.listdir(IMG_DIR + "strawberry_whites/train/")
strawberry_test_files = os.listdir(IMG_DIR + "strawberry_whites/test/")


def boxed_strawberry(img, json_file):
    berry = preprocess_image_train(img)
    berry_flat = tf.reshape(berry, [196608])
    x = json_file["x"]
    y = json_file["y"]
    width = json_file["width"]
    height = json_file["height"]
    bbox = [x, y, width, height]
    bbox = tf.convert_to_tensor(bbox, dtype="float32")
    berry_annotated = tf.concat([berry_flat, bbox], axis=0)

    return berry_annotated


# Define the datasets
train_strawberry = tf.data.Dataset.list_files(IMG_DIR + "strawberry_whites/train/*.png")
test_strawberry = tf.data.Dataset.list_files(IMG_DIR + "strawberry_whites/test/*.png")
train_weight = tf.data.Dataset.from_tensor_slices([np.load(elem)
                                                   for elem in getListOfFiles(IMG_DIR + "new_normalized_weight_images_splitted/train/")])
test_weight = tf.data.Dataset.from_tensor_slices([np.load(elem)
                                                  for elem in getListOfFiles(IMG_DIR + "new_normalized_weight_images_splitted/test/")])


# def load -> preprocessing.py

''' Preprocessing Pipeline '''
# def random_crop -> preprocessing.py
# def normalize -> preprocessing.py
# def random_jitter -> preprocessing.py
# def preprocess_image_train -> preprocessing.py
# def preprocess_image_test -> preprocessing.py

"""
    NEXT LINES OF CODE FUNCTIONS:
    - The tf.data.Dataset.cache transformation can cache a dataset, either in memory or on local storage.
      This will save some operations (like file opening and data reading) from being executed during each epoch.
      The next epochs will reuse the data cached by the cache transformation.
    - 'map' maps 'map_func'=preprocess_image_train across the elements of the dataset --> it applies the preprocessing
      function to each element of the dataset, returning a new preprocessed dataset. 'num_parallel_calls' often improves
      the performance, making 'map' use multiple threads (fili) to process elements.
    - 'random' randomly shuffles the elements of this dataset. This dataset fills a buffer with 'buffer_size' elements,
      then randomly samples elements from this buffer, replacing the selected elements with new elements. 
      For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required. 
    - 'batch' combines consecutive elements of the dataset in batches.
"""

train_strawberry = train_strawberry.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE)\
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train_weight = train_weight.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_strawberry = test_strawberry.cache().map(preprocess_image_test, num_parallel_calls=AUTOTUNE)\
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_weight = test_weight.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

'''
The iter() method returns an iterator for the given argument.
The next() function returns the next item from the iterator.
'''
sample_strawberry = next(iter(train_strawberry))
sample_weight = next(iter(train_weight))

# Plotting a strawberry to see the results of the random jittering operation
""""""
plt.subplot(121)
plt.title('Strawberry')
plt.imshow(sample_strawberry[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Strawberry with random jitter')
plt.imshow(random_jitter(sample_strawberry[0] * 0.5 + 0.5))
plt.show()

# Plotting a weight image to see the results of the random jittering operation
plt.subplot(121)
plt.title('Weight')
plt.imshow(sample_weight[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Weight')
plt.imshow(random_jitter(sample_weight[0] * 0.5 + 0.5))
plt.show()

""" End of Input Pipeline """

"""
Architecture generation pipeline.
    X == strawberry 
    Y == weights_dir
    Lets now create the architecture:
        generator G learns to transform image X to image Y
        generator F learns to transform image Y to image X
        discriminator D_X learns to differentiate between image X and generated image X -> strawberries     
        discriminator D_Y learns to differentiate between image Y and generated image Y -> weights_dir 
"""

OUTPUT_CHANNELS = 3

generator_g = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = discriminator(norm_type='instancenorm', target=False)
discriminator_y = discriminator(norm_type='instancenorm', target=False)

# A PART WITH JUST PLOTS OF SOME PREPROCESSED PICTURES HAS BEEN REMOVED
# IF NEEDED, GET IT FROM TENSORFLOW CYCLEGAN IMPLEMENTATION

""" Defining the Loss Functions """

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# def discriminator_loss -> model.py
# def generator_loss -> model.py
# def calc_cycle_loss -> model.py
# def identity_loss -> model.py


''' Initialize the optimizer for all the generators and the discriminators '''
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

""" Checkpoints """
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,  # Manages saving/restoring trackable values to disk
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)  # saves max 5 checkpoint

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

""" 
Training loop (in the paper was done for 200 epochs).
It consists of four basic steps:
    - get the predictions
    - calculate the loss
    - calculate the gradients using backpropagation 
    - apply the gradients to the optimizer 
"""

EPOCHS = 150


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.now().strftime("%d%m%y-%H%M"))


@tf.function
def train_step(real_x, real_y, epoch):      # bb_x, bb_y, bb_width, bb_height
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    # X == strawberry
    # Y == weight
    # Generator G translates X -> Y (strawberry to weight)
    # Generator F translates Y -> X (weight to strawberry)
    # Discriminator X discriminates between real and fake strawberry images
    # Discriminator Y discriminates between real and fake weight images

    # auxiliar_img = np.zeros([256, 256])
    # auxiliar_img[bb_y:(bb_y + bb_height), bb_x:(bb_x + bb_width)] = 1
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_x_loss(disc_real_x, disc_fake_x)

        disc_y_loss = discriminator_y_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_g_total_loss', total_gen_g_loss, step=epoch)
        tf.summary.scalar('gen_f_total_loss', total_gen_f_loss, step=epoch)
        tf.summary.scalar('disc_x_total_loss', disc_x_loss, step=epoch)
        tf.summary.scalar('disc_y_total_loss', disc_y_loss, step=epoch)


"""Execution of the training only when script directly executed"""
if __name__ == "__main__":

    for epoch in range(EPOCHS):
        start = time.time()

        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_strawberry, train_weight)):
            train_step(image_x, image_y, epoch)
            if n % 10 == 0:
                print('*', end='')
            n += 1

        clear_output(wait=True)  # Used to clear the previous figure. In case we generate_images every epoch.
        # Using a consistent image (sample_strawberry) so that the progress of the model
        # is clearly visible.
        generate_images(generator_g, sample_strawberry)

        if (epoch + 1) % 5 == 0:
            # We are saving checkpoints every 5 epochs. This takes up a lot of memory. Change if not necessary.
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))

        """ Lets save the trained models """
        generator_g.save(MODEL_GEN_G_PATH)
        generator_f.save(MODEL_GEN_F_PATH)
        discriminator_x.save(MODEL_DISC_X_PATH)
        discriminator_y.save(MODEL_DISC_Y_PATH)

    """ Generate using test dataset """
    # Run the trained model on the test dataset
    for inp in test_strawberry.take(5):
        generate_images(generator_g, inp)
