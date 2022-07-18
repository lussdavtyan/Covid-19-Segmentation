import numpy as np
import cv2

def int_over_union(lung, inf):
    lung = np.uint8(lung)
    inf = np.uint8(inf)
    union = cv2.bitwise_or(lung, inf)
    if np.sum(union) == 0:
        return 0
        
    return np.sum(inf) / np.sum(union)

def int_over_lung_area(lung, inf):
    lung = np.uint8(lung)
    inf = np.uint8(inf)
    if np.sum(lung) == 0:
        return 0

    return np.sum(inf) / np.sum(lung)

def dice_coef(y_true, y_pred):
    y_true = np.uint8(y_true)
    y_pred = np.uint8(y_pred)
    intersection = np.sum(cv2.bitwise_and(y_true, y_pred))
    if (np.sum(y_true) == 0) and (np.sum(y_pred) == 0):
        return 1

    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_recall(true_img, test_img):
    tp = cv2.bitwise_and(true_img, test_img)
    if np.sum(true_img) == 0:
        recall = 0.0
    else:
        recall = np.sum(tp) / np.sum(true_img)

    return recall

def get_precision(true_img, test_img):
    tp = cv2.bitwise_and(true_img, test_img)
    if np.sum(test_img) == 0:
        precision = 0.0
    else:
        precision = np.sum(tp) / np.sum(test_img)

    return precision
