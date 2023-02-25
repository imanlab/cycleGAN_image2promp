import inspect
import os

from datetime import datetime


today = datetime.now().strftime("%d_%m_%y")
now = datetime.now().strftime("%d_%m_%y__%H_%M")

# MAIN ROOT
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dataset_dir = os.path.dirname(os.path.dirname(current_dir)) + "/dataset/"
print("la current directory è:", current_dir)
print("la dataset directory è:", dataset_dir)
# LOSS PLOT DIRECTORY
PLOT_PATH = current_dir + "/LOSS"
# IMG DIRECTORY
IMG_DIR = dataset_dir + "dataset_pca/"
# MODEL SAVE DIRECTORY
MODEL_GEN_G_PATH = current_dir + "/MODEL_GEN_G_" + today
MODEL_GEN_F_PATH = current_dir + "/MODEL_GEN_F_" + today
MODEL_DISC_X_PATH = current_dir + "/MODEL_DISC_X_" + today
MODEL_DISC_Y_PATH = current_dir + "/MODEL_DISC_Y_" + today

# RECONSTRUCTED IMAGES DIRECTORY
IMAGES = current_dir + "/RECONSTRUCTED IMAGES"

"""Following lines till the end imported from file 'config_J1.py' chat Teams with Chiara"""
# Experiment directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))     # /home/francesco

# RGB-D images folder.
IMAGE_PATH = os.path.join(ROOT_DIR, "PycharmProjects/dataset/dataset_pca/strawberry_imgs/")

# Annotations (mean and covariance) folder.
ANNOTATION_PATH = os.path.join(ROOT_DIR, "PycharmProjects/dataset/dataset_pca/probabilistic_7dof/")

# Annotations (BB == boundary box) folder.
PATH_TO_BB = "/home/francesco/PycharmProjects/dataset/dataset_pca/annotation/reshaped_json/"
