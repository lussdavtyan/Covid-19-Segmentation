from nibabel.testing import data_path
import nibabel as nib
import cv2
import numpy as np
import kornia, skimage
import matplotlib.pyplot as plt
import os
import math
import argparse
from os import path
import re   
import util_functions
import segmentation_parameters
import segmentation_functions

parser = argparse.ArgumentParser()

parser.add_argument('--img_path', type=str, required=True, help='Path to image data')
args = parser.parse_args()

img_path = args.img_path

img_file_paths = [f for f in os.listdir(img_path) if re.match(r'^coronacases', f)]

hu_img_list = []

for path in img_file_paths:
    img = nib.load(os.path.join(img_path, path))
    image_data = img.get_fdata() 
    image_data = np.moveaxis(image_data, -1, 0)  
    hu_images = []
    for image in image_data:
        hu_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    hu_images = np.array(hu_images)
    hu_img_list.append(hu_images)

segmented_lungs_list = []
for hu_images in hu_img_list:
    segmented_lungs = [segmentation_functions.segment_lungs(image) for image in hu_images]
    segmented_lungs = np.array(segmented_lungs)
    segmented_lungs_list.append(segmented_lungs)

segmented_rib_cage_list = []
for hu_images in hu_img_list:
    segmented_rib_cage_list.append(segmentation_functions.segment_rib_cage(hu_images))
    


thresholded_superpixels_list = []
for inside_rib_cage_slices in segmented_rib_cage_list:
    thresholded_list = []
    for inside_rib_cage in inside_rib_cage_slices:
        labels, points, means, sds = segmentation_functions.extract_superpixels(inside_rib_cage)
        thresholded_superpixels = segmentation_functions.threshold_superpixels(labels, points, means, sds)
        thresholded_list.append(thresholded_superpixels)
    thresholded_superpixels_list.append(thresholded_list)


