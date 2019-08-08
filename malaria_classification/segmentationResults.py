#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:54:44 2019

@author: reganlamoureux
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import os
import skimage
import skimage.filters as flt
import skimage.morphology as morph

from scipy import ndimage as ndi
from skimage import measure
from skimage.color import rgb2gray
#%matplotlib inline

counter = 1

def openCV(path):
    img = cv2.imread('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution/'+path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    filled = ndi.binary_fill_holes(thresh)
    filled = np.uint8(filled)

    # sure background area
    sure_bg = cv2.dilate(filled,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(filled,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==1] = 0
    markers_1 = markers.copy()
    ws_img = cv2.watershed(img, markers)
    img[ws_img == -1] = [255,0,0]
    
    plt.imsave('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution/openCV/watershed/{}png'.format(path[:-3]), ws_img)
    plt.imsave('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution/openCV/original w lines/{}png'.format(path[:-3]), img)
    return ret

def sk_image(path):
    img = skimage.io.imread('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution/'+path)
    img_gs = rgb2gray(img)
    thr = flt.threshold_otsu(img_gs)
    thresh_img = img_gs < thr

    kernel = np.ones((3,3),np.uint8)
    filled = ndi.binary_fill_holes(thresh_img)
    filled = np.uint8(filled)


    # sure background area
    sure_bg = morph.dilation(filled, kernel)

    # Finding sure foreground area
    dist_transform = ndi.distance_transform_edt(filled)
    dist_thr = int(0.5*dist_transform.max())
    sure_fg = dist_transform > dist_thr # cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)


    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = sure_bg - sure_fg

    markers, num_labels = measure.label(sure_fg, background=0, return_num=True)
    # smooth img_gs to remove noise
    img_smth = flt.gaussian(img_gs, sigma=5)
    # edgemap
    edg_img = flt.sobel(img_smth)
    ws_img = morph.watershed(edg_img, markers)
    plt.imsave('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution/skimage/{}png'.format(path[:-3]), ws_img)
    return num_labels

def excel(sheetName, imageName, ret):
    xfile = openpyxl.load_workbook('Segmentation Results.xlsx')
    xfile.sheetnames # all names
    sheet = xfile[sheetName]
    nameLine = 'A' + str(counter)
    sheet[nameLine] = imageName
    
    numLine = 'B' + str(counter)
    sheet[numLine] = ret
    xfile.save('Segmentation Results.xlsx')
    
def image_to_excel(my_path):
    global counter
    folder = os.fsdecode(my_path)
    #return os.listdir(folder)
    #for image in os.listdir(folder):
    for image in os.listdir(folder):
        #print(str(counter) + '. ' + image)
        if image=='.DS_Store':
            pass
        else:
            ret = openCV(image)
            excel("OpenCV", image, ret)
            num_labels = sk_image(image)
            excel("Skimage", image, num_labels)
            counter += 1

image_to_excel('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution')


    
