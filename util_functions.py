import cv2
import numpy as np
import math
import re
import cc3d
import networkx as nx
import segmentation_parameters

def threshold_by_three_levels(extracted_lung_slice, t1, t2):
    _, thresholded_infection1 = cv2.threshold(np.int16(extracted_lung_slice), t1, 255, cv2.THRESH_BINARY)
    _, thresholded_infection2 = cv2.threshold(np.int16(extracted_lung_slice), t2, 255, cv2.THRESH_BINARY_INV)
    thresholded_infection = cv2.bitwise_and(thresholded_infection1, thresholded_infection2)
    return thresholded_infection


def remove_small_components(img, min_size):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(img), connectivity=8)
    component_sizes = stats[1:, -1]

    large_components = np.zeros((labels.shape))
    for i in range(num_labels - 1):
        if component_sizes[i] >= min_size:
            large_components[labels == i + 1] = 255
    return large_components


def compute_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def find_closest_to_point(x, y, lst):
    min_distance = 100000
    closest_x = -1
    closest_y = -1
    for i in range(len(lst)):
        curr_dist = compute_distance(x, y, lst[i][0], lst[i][1])
        if(curr_dist < min_distance):
            min_distance = curr_dist
            closest_x = lst[i][0]
            closest_y = lst[i][1]

    return closest_x, closest_y, min_distance


def find_closest_to_contour(lst1, lst2):
    min_distance = 100000
    x1, y1, x2, y2 = -1, -1, -1, -1
    for i in range(len(lst1)):
        curr_x, curr_y, curr_dist = find_closest_to_point(lst1[i][0], lst1[i][1], lst2)
        if curr_dist < min_distance:
            min_distance = curr_dist
            x1 = lst1[i][0]
            y1 = lst1[i][1]
            x2 = curr_x
            y2 = curr_y

    return x1, y1, x2, y2, min_distance


def fill_the_holes(image):
    img = np.uint8(image)
    contours = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]
    greatest_connected_component = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.drawContours(image=greatest_connected_component, contours=contours,
                    contourIdx=-1, color=(255, 255, 255), thickness=-1)
    return greatest_connected_component


def add_vertical_lines(image):
    image = cv2.line(np.uint8(image), (150, 0), (200, image.shape[0]), color=0, thickness=1)
    image = cv2.line(np.uint8(image), (200, 0), (200, image.shape[0]), color=0, thickness=1)
    image = cv2.line(np.uint8(image), (250, 0), (300, image.shape[0]), color=0, thickness=1)
    image = cv2.line(np.uint8(image), (300, 0), (300, image.shape[0]), color=0, thickness=1)
    image = cv2.line(np.uint8(image), (350, 0), (300, image.shape[0]), color=0, thickness=1)
    return image


def add_horizontal_lines(image):
    image = cv2.line(np.uint8(image), (0, 125), (image.shape[1], 125), color=0, thickness=1)
    image = cv2.line(np.uint8(image), (0, 165), (image.shape[1], 165), color=0, thickness=1)
    image = cv2.line(np.uint8(image), (0, 195), (image.shape[1], 165), color=0, thickness=1)
    image = cv2.line(np.uint8(image), (0, 225), (image.shape[1], 165), color=0, thickness=1)
    return image