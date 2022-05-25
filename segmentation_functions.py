import cv2
import numpy as np
import math
import re
import cc3d
import networkx as nx
import segmentation_parameters
import util_functions

def segment_lungs(img):
    _, thresholded = cv2.threshold(img, segmentation_parameters.lung_segmentation_threshold, 255, cv2.THRESH_BINARY) 
    thresholded = np.uint8(thresholded)
    body_mask = get_body_mask(img)
    thresholded_inverse = cv2.bitwise_not(thresholded)
    segmented_lung = cv2.bitwise_and(thresholded_inverse, body_mask)

    return util_functions.fill_the_holes(segmented_lung)


def get_body_mask(img):
    _, thresholded = cv2.threshold(img, segmentation_parameters.lung_segmentation_threshold, 255, cv2.THRESH_BINARY) 

    thresholded = np.uint8(thresholded)
    hole_filled = util_functions.fill_the_holes(np.uint8(thresholded))

    greatest_connected_component = np.zeros((hole_filled.shape[0], hole_filled.shape[1]), np.uint8)

    contours = cv2.findContours(image=hole_filled.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(image=greatest_connected_component, contours=[max(contours, key = cv2.contourArea)],
                    contourIdx=0, color=(255, 255, 255), thickness=-1)

    return greatest_connected_component


def unite_consecutive_slices(segmented_ribs, number_of_slices):
    segmented_ribs_union = []
    for i in range(len(segmented_ribs)):
        img = segmented_ribs[i]
        for j in range(1, number_of_slices):
            if i + j >= len(segmented_ribs):
                break
            img = cv2.bitwise_or(img, segmented_ribs[i + j])
        segmented_ribs_union.append(img)
    segmented_ribs_union = np.array(segmented_ribs_union)

    return segmented_ribs_union


def remove_scapula(segmented_ribs):
    labels_out = cc3d.largest_k(
        segmented_ribs, k=1,
        connectivity=26, delta=0
    )

    segmented_ribs *= (labels_out > 0)

    return segmented_ribs


def get_extreme_points(segmented_ribs_union, connected_components_stats):
    extreme_points_all_slices = []
    for k in range(len(segmented_ribs_union)):
        num_labels, labels, _, _ = connected_components_stats[k]
        points = []
        for i in range(num_labels):
            points.append([])
        for i in range(len(segmented_ribs_union[k])):
            for j in range(len(segmented_ribs_union[k][i])):
                label = labels[i][j]
                points[label].append((j, i))
        extreme_points_all = np.zeros((num_labels, 4, 2))
        for i in range(len(points)):
            component_points = np.array(points[i])
            extreme_points = []
            extreme_points.append(component_points[component_points[:, 0].argmin()])
            extreme_points.append(component_points[component_points[:, 0].argmax()])
            extreme_points.append(component_points[component_points[:, 1].argmin()])
            extreme_points.append(component_points[component_points[:, 1].argmax()])
            extreme_points = np.array(extreme_points)
            extreme_points_all[i] = extreme_points
        
        extreme_points_all_slices.append(extreme_points_all)

    return extreme_points_all_slices


def construct_mst(segmented_ribs_union, connected_components_stats):
    mst_list = []

    for k in range(len(segmented_ribs_union)):
        num_labels, _, _, centroids = connected_components_stats[k]
        n = num_labels - 1
        graph = nx.Graph()

        c = 1
        for i in range(1, len(centroids)):
            graph.add_node(c, px=(centroids[i][0], centroids[i][1]))
            c += 1

        for x in graph.nodes:
            for y in graph.nodes:
                dist = util_functions.compute_distance(graph.nodes[x]['px'][0], graph.nodes[x]['px'][1],
                                                    graph.nodes[y]['px'][0], graph.nodes[y]['px'][1])
                graph.add_edge(x, y, weight=dist)

        mst = nx.minimum_spanning_tree(graph)
        mst_list.append(mst)

    return mst_list


def complete_the_cycle(mst_list):
    for i in range(len(mst_list)):
        root = list(mst_list[i].nodes)[0]
        edges = nx.bfs_edges(mst_list[i], root)
        nodes = [root] + [v for u, v in edges]
        edges = nx.bfs_edges(mst_list[i], nodes[-1])
        nodes = [nodes[-1]] + [v for u, v in edges]
        mst_list[i].add_edge(nodes[0], nodes[-1])

    return mst_list


def connect_extreme_points(segmented_ribs_union, mst_list, extreme_points_all_slices):
    connected_extreme_points_from_centroids_mst = segmented_ribs_union
    for i in range(len(mst_list)):
        for edge in mst_list[i].edges:
            node1 = edge[0]
            node2 = edge[1]
            extreme_points1 = np.array(extreme_points_all_slices[i][node1])
            extreme_points2 = np.array(extreme_points_all_slices[i][node2])

            x1, y1, x2, y2, _ = util_functions.find_closest_to_contour(extreme_points1, extreme_points2)
            connected_extreme_points_from_centroids_mst[i] = cv2.line(connected_extreme_points_from_centroids_mst[i],
                                                                    (int(x1), int(y1)), (int(x2), int(y2)), color=255,
                                                                    thickness=2)

    return connected_extreme_points_from_centroids_mst


def fill_inside_rib_cage(hu_images, connected_extreme_points_from_centroids_mst):
    inside_rib_cage = []
    for i in range(len(connected_extreme_points_from_centroids_mst)):
        flood_filled = connected_extreme_points_from_centroids_mst[i].copy()
        mean_x = -1
        mean_y = -1
        count = 0

        for j in range(len(flood_filled)):
            for k in range(len(flood_filled[j])):
                if connected_extreme_points_from_centroids_mst[i][j][k] > 0:
                    mean_x += k
                    mean_y += j
                    count += 1

        mean_x /= count
        mean_y /= count

        while connected_extreme_points_from_centroids_mst[i][int(mean_y)][int(mean_x)] > 0:
            mean_y -= 1

        x = flood_filled.shape[1] / 2
        y = flood_filled.shape[0] / 3
        x = mean_x
        y = mean_y
        cv2.floodFill(image=flood_filled, mask=None, seedPoint=(int(x), int(y)), newVal=200)
        extracted = np.where(flood_filled == 200, hu_images[i], -1024)
        inside_rib_cage.append(extracted)

    inside_rib_cage = np.array(inside_rib_cage)
    
    return inside_rib_cage

    
def segment_rib_cage(hu_images):
    body_masks = [get_body_mask(image) for image in hu_images]
    body_masks = np.array(body_masks)

    extracted_bodies = [np.where(body_mask, hu_image, -1024) for body_mask, hu_image in
                        zip(body_masks, hu_images)]  
    extracted_bodies = np.array(extracted_bodies)

    segmented_ribs = [cv2.threshold(image, segmentation_parameters.ribs_segmentation_threshold, 255, cv2.THRESH_BINARY)[1] for image in
                    extracted_bodies]

    segmented_ribs = np.array(segmented_ribs)
    segmented_ribs = remove_scapula(segmented_ribs)

    segmented_ribs_union = unite_consecutive_slices(segmented_ribs, segmentation_parameters.slices_in_union)

    segmented_ribs_union = [util_functions.fill_the_holes(image) for image in segmented_ribs_union]
    segmented_ribs_union = [util_functions.add_vertical_lines(image) for image in segmented_ribs_union]
    segmented_ribs_union = [util_functions.add_horizontal_lines(image) for image in segmented_ribs_union]

    segmented_ribs_union = np.array(segmented_ribs_union)

    connected_components_stats = [cv2.connectedComponentsWithStats(np.uint8(image)) for image in segmented_ribs_union]

    extreme_points_all_slices = get_extreme_points(segmented_ribs_union, connected_components_stats)
    mst_list = construct_mst(segmented_ribs_union, connected_components_stats)
    mst_list = complete_the_cycle(mst_list)

    connected_extreme_points_from_centroids_mst = connect_extreme_points(segmented_ribs_union, mst_list, extreme_points_all_slices)
    inside_rib_cage = fill_inside_rib_cage(hu_images, connected_extreme_points_from_centroids_mst)

    return inside_rib_cage


def extract_superpixels(inside_rib_cage):
    converted_img = np.uint8(inside_rib_cage)
    height, width = converted_img.shape
    channels = 1
    num_iterations = 6
    prior = 2
    num_levels = 1
    num_histogram_bins = 5

    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, segmentation_parameters.num_superpixels, num_levels, prior, num_histogram_bins)
    color_img = np.zeros((height, width, 1), converted_img.dtype)
    color_img[:] = 255
    seeds.iterate(converted_img, num_iterations)
    labels = seeds.getLabels()

    number_of_superpixels = seeds.getNumberOfSuperpixels()

    points = []
    means = [0 for i in range(number_of_superpixels)]

    for i in range(number_of_superpixels): 
        points.append([])

    for i in range(len(labels)):
        for j in range(len(labels[i])):
            label = labels[i][j]
            means[label] += inside_rib_cage[i][j]
            points[label].append((i, j))

    for i in range(0,len(points)):
        means[i] /= len(points[i])

    sds = [0 for i in range(number_of_superpixels)]

    for i in range(len(labels)):
        for j in range(len(labels[i])):
            label = labels[i][j]
            mean = means[label]
            n = len(points[i])
            sd = math.sqrt(((inside_rib_cage[i][j] - mean)**2) / n)
            sds[label] = sd
    return labels, points, means, sds

def threshold_superpixels(labels, points, means, sds):
    thresholded_superpixels = np.full((labels.shape[0], labels.shape[1]), -1024)
    for i in range(len(means)):
        if means[i] > segmentation_parameters.infection_segmentation_threshold_left and means[i] < segmentation_parameters.infection_segmentation_threshold_right and sds[i] < segmentation_parameters.standard_deviation_threshold:
            for j in range(len(points[i])):
                p = points[i][j]
                thresholded_superpixels[p[0]][p[1]] = 3071
    return thresholded_superpixels