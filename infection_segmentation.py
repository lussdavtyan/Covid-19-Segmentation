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

inf_proportion_list_gt1 = []
inf_proportion_list_pr1 = []
inf_proportion_list_gt2 = []
inf_proportion_list_pr2 = []
dice_list_lungs = []
jaccard_list_lungs = []
recall_list = []
precision_list = []
dice_list_inf = []
jaccard_list_inf = []

for path in img_file_paths:
    inf_proportion_list_gt1_ = []
    inf_proportion_list_pr1_ = []
    inf_proportion_list_gt2_ = []
    inf_proportion_list_pr2_ = []
    dice_list_lungs_ = []
    jaccard_list_lungs_ = []
    recall_list_ = []
    precision_list_ = []
    dice_list_inf_ = []
    jaccard_list_inf_ = []

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
        dice_list_lungs.append(dice)
        jaccard_list_lungs.append(jaccard)
        dice_list_lungs_.append(dice)
        jaccard_list_lungs_.append(jaccard)
        
    for im_true, im_test in zip(lung_masks, segmented_rib_cage):
        im_true = im_true.astype('int64')
        recall = performance_metrics.get_recall(im_true, im_test)
        precision = performance_metrics.get_precision(im_true, im_test)
        precision_list.append(precision)
        recall_list.append(recall)
        precision_list_.append(precision)
        recall_list_.append(recall)
        
    for im_true, im_test in zip(inf_masks, thresholded):
        dice = performance_metrics.dice_coef(im_true, im_test)
        jaccard = jaccard_score(im_true, im_test, average='micro')
        dice_list_inf.append(dice)
        jaccard_list_inf.append(jaccard)
        dice_list_inf_.append(dice)
        jaccard_list_inf_.append(jaccard)
        
    for im_true, im_test in zip(inf_masks, lung_masks):
        iou = performance_metrics.int_over_union(im_true, im_test)
        inf_proportion_list_gt1.append(iou)
        inf_proportion_list_gt1_.append(iou)
        iou = performance_metrics.int_over_lung_area(im_true, im_test)
        inf_proportion_list_gt2.append(iou)
        inf_proportion_list_gt2_.append(iou)
        

    for im_true, im_test in zip(thresholded, segmented_lungs):
        iou = performance_metrics.int_over_union(im_true, im_test)
        inf_proportion_list_pr1.append(iou)   
        inf_proportion_list_pr1_.append(iou)   
        iou = performance_metrics.int_over_lung_area(im_true, im_test)
        inf_proportion_list_pr2.append(iou)
        inf_proportion_list_pr2_.append(iou)
        
    dice_list_lungs_ = np.array(dice_list_lungs_)
    jaccard_list_lungs_ = np.array(jaccard_list_lungs_)
    precision_list_ = np.array(precision_list_)
    recall_list_ = np.array(recall_list_)
    dice_list_inf_ = np.array(dice_list_inf_)
    jaccard_list_inf_ = np.array(jaccard_list_inf_)
    rmse1_ = performance_metrics.rmse(np.array(inf_proportion_list_pr1_), np.array(inf_proportion_list_gt1_))
    rmse2_ = performance_metrics.rmse(np.array(inf_proportion_list_pr2_), np.array(inf_proportion_list_gt2_))

    print(path)
    print("Lung segmentation evaluation")
    print("Dice: ", np.mean(dice_list_lungs_), "Jaccard: ", np.mean(jaccard_list_lungs_))
    print("Rib cage segmentation evaluation")
    print("Precision: ", np.mean(precision_list_), "Recall: ", np.mean(recall_list_))
    print("Infection segmentation evaluation")
    print("Dice: ", np.mean(dice_list_inf_), "Jaccard: ", np.mean(jaccard_list_inf_))
    print("Infection quantification evaluation")
    print("version1: ", rmse1_, "version2: ", rmse2_)
    print()


dice_list_lungs = np.array(dice_list_lungs)
jaccard_list_lungs = np.array(jaccard_list_lungs)
precision_list = np.array(precision_list)
recall_list = np.array(recall_list)
dice_list_inf = np.array(dice_list_inf)
jaccard_list_inf = np.array(jaccard_list_inf)
rmse1 = performance_metrics.rmse(np.array(inf_proportion_list_pr1), np.array(inf_proportion_list_gt1))
rmse2 = performance_metrics.rmse(np.array(inf_proportion_list_pr2), np.array(inf_proportion_list_gt2))    


print("\nAverage results of all cases")
print("Lung segmentation evaluation")
print("Dice: ", np.mean(dice_list_lungs), "Jaccard: ", np.mean(jaccard_list_lungs))
print("Rib cage segmentation evaluation")
print("Precision: ", np.mean(precision_list), "Recall: ", np.mean(recall_list))
print("Infection segmentation evaluation")
print("Dice: ", np.mean(dice_list_inf), "Jaccard: ", np.mean(jaccard_list_inf))
print("Infection quantification evaluation")
print("version1: ", rmse1, "version2: ", rmse2)
