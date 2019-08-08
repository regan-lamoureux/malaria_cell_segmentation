import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import openpyxl


from scipy import ndimage as ndi
from skimage import measure, feature

counter = 1

def feature_extraction(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eight_bit = np.uint8(gray)

    cm = feature.greycomatrix(eight_bit, [1], [0])
    
    
    contrast = feature.greycoprops(cm)
    dissimilarity = feature.greycoprops(cm, 'dissimilarity')
    homogeneity = feature.greycoprops(cm, 'homogeneity')
    ASM = feature.greycoprops(cm, 'ASM')
    energy = feature.greycoprops(cm, 'energy')
    correlation = feature.greycoprops(cm, 'correlation')
    ent = measure.shannon_entropy(cm)

    return [contrast[0][0], dissimilarity[0][0], homogeneity[0][0], ASM[0][0], energy[0][0], correlation[0][0], ent]

def eccentricity(path):
    image_rgb = skimage.io.imread(path)

    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=3,
              low_threshold=10, high_threshold=50)
    thr = flt.threshold_otsu(image_gray)
    img = image_gray < thr
    filled = scipy.ndimage.morphology.binary_fill_holes(img)
    label = measure.label(filled)

    for cell in measure.regionprops(label):
        return cell.eccentricity


def excel(workbook, imageName, feat_list):
    global counter
    xfile = openpyxl.load_workbook(workbook)
    xfile.sheetnames  # all names
    sheet = xfile['Sheet1']


    for i in range(0, len(feat_list)):
        line = alphabet[i] + str(counter)
        sheet[line] = feat_list[i]

    xfile.save(workbook)

def image_to_excel(my_path, workbook):
    global counter
    folder = os.fsdecode(my_path)
    # return os.listdir(folder)
    # for image in os.listdir(folder):
    for i in range(0, 344):
        # print(str(counter) + '. ' + image)
        if os.listdir(folder)[i] == '.DS_Store':
            pass
        else:
            ret = feature_extraction(my_path + '/' + os.listdir(folder)[i])
            excel(workbook, os.listdir(folder)[i], ret)
            counter += 1



image_to_excel('/Users/reganlamoureux/troph', 'troph_data.xlsx')