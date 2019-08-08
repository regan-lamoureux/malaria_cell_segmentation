#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:36:07 2019

@author: reganlamoureux
"""

import cv2
import numpy as np
import os

from matplotlib import pyplot as plt
from scipy import ndimage as ndi

def watershed(image):
    img = cv2.imread('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution/New Folder With Items/'+image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    kernel = np.ones((3,3),np.uint8)
    filled = ndi.binary_fill_holes(thresh)
    filled = np.uint8(filled)

    # sure background area
    sure_bg = cv2.dilate(filled,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(filled,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==1] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255,0,0]

    


