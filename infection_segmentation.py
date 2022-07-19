import numpy as np
import os
import argparse
from os import path
import re
from sklearn.metrics import jaccard_score
import util_functions
import segmentation_functions
import performance_metrics


parser = argparse.ArgumentParser()

parser.add_argument('--img_path', type=str, required=True, help='Path to image data')
parser.add_argument('--lung_mask_path', type=str, required=True, help='Path to lung mask data')
parser.add_argument('--inf_mask_path', type=str, required=True, help='Path to infection mask data')
args = parser.parse_args()


img_path = args.img_path
lung_mask_path = args.lung_mask_path
inf_mask_path = args.inf_mask_path

img_file_paths = [f for f in os.listdir(img_path) if re.match(r'^coronacases', f)]
lung_mask_file_paths = [f for f in os.listdir(img_path) if re.match(r'^coronacases', f)]
inf_mask_file_paths = [f for f in os.listdir(img_path) if re.match(r'^coronacases', f)]

inf_proportion_gt1 = np.array([])
inf_proportion_pr1 = np.array([])
inf_proportion_gt2 = np.array([])
inf_proportion_pr2 = np.array([])
dice_lungs = np.array([])
jaccard_lungs = np.array([])
recall_rib_cage = np.array([])
precision_rib_cage = np.array([])
dice_infection = np.array([])
jaccard_infection = np.array([])

for path in img_file_paths:
    inf_proportion_gt1_ = np.array([])
    inf_proportion_pr1_ = np.array([])
    inf_proportion_gt2_ = np.array([])
    inf_proportion_pr2_ = np.array([])
    dice_lungs_ = np.array([])
    jaccard_lungs_ = np.array([])
    recall_rib_cage_ = np.array([])
    precision_rib_cage_ = np.array([])
    dice_infection_ = np.array([])
    jaccard_infection_ = np.array([])

    hu_images = util_functions.load_nii_files(os.path.join(img_path, path))
    lung_masks = util_functions.load_nii_files(os.path.join(lung_mask_path, path))
    for i in range(len(lung_masks)):
        lung_masks[i] = np.where(lung_masks[i] == 0, 0, 1)
    inf_masks = util_functions.load_nii_files(os.path.join(inf_mask_path, path))
    
    segmented_lungs = segmentation_functions.segment_lungs(hu_images)    
    segmented_rib_cage = segmentation_functions.segment_rib_cage(hu_images)
    inside_rib_cage = np.where(segmented_rib_cage == 0, -1024, hu_images)

    thresholded = []
    for irc in inside_rib_cage:
        labels, points, means, sds = segmentation_functions.extract_superpixels(irc)
        thresholded_superpixels = segmentation_functions.threshold_superpixels(labels, points, means, sds)
        thresholded_superpixels = np.where(thresholded_superpixels == -1024, 0, 1)
        thresholded.append(thresholded_superpixels)
    
    for im_true, im_test in zip(lung_masks, segmented_lungs):
        dice = performance_metrics.dice_coef(im_true, im_test)
        jaccard = jaccard_score(im_true, im_test, average='micro')
        dice_lungs = np.append(dice_lungs, dice)
        jaccard_lungs = np.append(jaccard_lungs, jaccard)
        dice_lungs_ = np.append(dice_lungs_, dice)
        jaccard_lungs_ = np.append(jaccard_lungs_, jaccard)
        
    for im_true, im_test in zip(lung_masks, segmented_rib_cage):
        im_true = im_true.astype('int64')
        recall = performance_metrics.get_recall(im_true, im_test)
        precision = performance_metrics.get_precision(im_true, im_test)
        precision_rib_cage = np.append(precision_rib_cage, precision)
        recall_rib_cage = np.append(recall_rib_cage, recall)
        precision_rib_cage_ = np.append(precision_rib_cage_, precision)
        recall_rib_cage_ = np.append(recall_rib_cage_, recall)
        
    for im_true, im_test in zip(inf_masks, thresholded):
        dice = performance_metrics.dice_coef(im_true, im_test)
        jaccard = jaccard_score(im_true, im_test, average='micro')
        dice_infection = np.append(dice_infection, dice)
        jaccard_infection = np.append(jaccard_infection, jaccard)
        dice_infection_ = np.append(dice_infection_, dice)
        jaccard_infection_ = np.append(jaccard_infection_, jaccard)
        
    for im_true, im_test in zip(inf_masks, lung_masks):
        iou = performance_metrics.int_over_union(im_true, im_test)
        inf_proportion_gt1 = np.append(inf_proportion_gt1, iou)
        inf_proportion_gt1_ = np.append(inf_proportion_gt1_, iou)
        iou = performance_metrics.int_over_lung_area(im_true, im_test)
        inf_proportion_gt2 = np.append(inf_proportion_gt2, iou)
        inf_proportion_gt2_ = np.append(inf_proportion_gt2_, iou)
        

    for im_true, im_test in zip(thresholded, segmented_lungs):
        iou = performance_metrics.int_over_union(im_true, im_test)
        inf_proportion_pr1 = np.append(inf_proportion_pr1, iou)   
        inf_proportion_pr1_ = np.append(inf_proportion_pr1_, iou)   
        iou = performance_metrics.int_over_lung_area(im_true, im_test)
        inf_proportion_pr2 = np.append(inf_proportion_pr2, iou)
        inf_proportion_pr2_ = np.append(inf_proportion_pr2_, iou)
        
   
    rmse1_ = performance_metrics.rmse(inf_proportion_pr1_, inf_proportion_gt1_)
    rmse2_ = performance_metrics.rmse(inf_proportion_pr2_, inf_proportion_gt2_)

    print(path)
    util_functions.print_evaluation_results(dice_lungs_, jaccard_lungs_, precision_rib_cage_, recall_rib_cage_, dice_infection_, jaccard_infection_, rmse1_, rmse2_)
    print()


rmse1 = performance_metrics.rmse(inf_proportion_pr1, inf_proportion_gt1)
rmse2 = performance_metrics.rmse(inf_proportion_pr2, inf_proportion_gt2)    


print("\nAverage results of all cases")
util_functions.print_evaluation_results(dice_lungs, jaccard_lungs, precision_rib_cage, recall_rib_cage, dice_infection, jaccard_infection, rmse1, rmse2)
