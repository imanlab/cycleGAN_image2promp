"""
This file is used to create the annotated strawberry image, where the selected strawberry is highlighted and noise added.
It was also used to reshape the coordinates of the annotation box from a 720x1280 img to a 256x256 img.
"""
import json
import os
import numpy as np
from PIL import Image

from config import IMG_DIR
from preprocessing import load_image
from image_average import noisy

path_to_annotation = "/home/francesco/PycharmProjects/dataset/dataset_pca/annotation"
path_to_json = "/home/francesco/PycharmProjects/dataset/dataset_pca/annotation/annotation_json/"
strawberry_dir = "/home/francesco/PycharmProjects/dataset/collected_dataset/dataset_reach_to_pick/all_config_imm/"
path_to_save = "/home/francesco/PycharmProjects/dataset/dataset_autoencoder/strawberry_bb"

if __name__ == "__main__":

    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    for file in json_files:
        f = open(path_to_json + file)
        segmentation = json.load(f)
        key = list(segmentation.keys())[0]
        x = int((segmentation[key]["regions"][0]["shape_attributes"]["x"]))
        y = int((segmentation[key]["regions"][0]["shape_attributes"]["y"]))
        width = int((segmentation[key]["regions"][0]["shape_attributes"]["width"]))
        height = int((segmentation[key]["regions"][0]["shape_attributes"]["height"]))
        filename = (segmentation[key]["filename"])

        strawberry_files = os.listdir(strawberry_dir)

        for strawberry in strawberry_files:
            berry = strawberry.split(".png")
            berry = berry[0][:-6]
            berry = berry + ".png"
            if filename == berry:
                image = Image.open(strawberry_dir + strawberry)
                pixel_map = image.load()
                w, h = image.size
                white = np.full((w, h, 3), 255, dtype="float32")
                white_noisy = noisy(white, "gauss").astype("uint8")
                for i in range(h):
                    for j in range(x):
                        r = white_noisy[j, i][0]
                        g = white_noisy[j, i][1]
                        b = white_noisy[j, i][2]
                        pixel_map[j, i] = (r, g, b)
                for i in range(y):
                    for j in range(w):
                        r = white_noisy[j, i][0]
                        g = white_noisy[j, i][1]
                        b = white_noisy[j, i][2]
                        pixel_map[j, i] = (r, g, b)
                for i in range(h):
                    for j in range(x+width, w):
                        r = white_noisy[j, i][0]
                        g = white_noisy[j, i][1]
                        b = white_noisy[j, i][2]
                        pixel_map[j, i] = (r, g, b)
                for i in range(y+height, h):
                    for j in range(w):
                        r = white_noisy[j, i][0]
                        g = white_noisy[j, i][1]
                        b = white_noisy[j, i][2]
                        pixel_map[j, i] = (r, g, b)
                        # pixel_map[j, i] = (int(256), int(256), int(256))

                image.save(IMG_DIR + "strawberry_whites_noise/" + strawberry.replace(".png", "_noise.png"))

        """
        NEXT LINES WERE USED TO RESHAPE THE BOUNDARY BOX FOR A 256X256 IMAGE
        
        x_reshaped = round(256 / 1280 * x)
        y_reshaped = round(256 / 720 * y)
        width_reshaped = round(256 / 1280 * width)
        height_reshaped = round(256 / 720 * height)

        dictionary = {
            "filename": segmentation[key]["filename"],
            "x": x_reshaped,
            "y": y_reshaped,
            "width": width_reshaped,
            "height": height_reshaped
        }
        
        with open(path_to_annotation + "reshaped_json/" + file, "w") as outfile:
            json.dump(dictionary, outfile)
        """
