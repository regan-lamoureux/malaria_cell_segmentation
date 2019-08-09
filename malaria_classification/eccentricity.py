#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:08:38 2019

@author: reganlamoureux
"""

import numpy as np
import scipy
import skimage
import skimage.filters as flt

from skimage import color
from skimage import feature
from skimage import measure




def feature_extraction(path):
    image_rgb = skimage.io.imread(path)

    image_gray = color.rgb2gray(image_rgb)
    
    eight_bit = np.uint8(image_gray)

    cm = feature.greycomatrix(eight_bit, [1], [0])
    
    
    contrast = feature.greycoprops(cm)
    dissimilarity = feature.greycoprops(cm, 'dissimilarity')
    homogeneity = feature.greycoprops(cm, 'homogeneity')
    ASM = feature.greycoprops(cm, 'ASM')
    energy = feature.greycoprops(cm, 'energy')
    correlation = feature.greycoprops(cm, 'correlation')
    ent = measure.shannon_entropy(cm)
    
    thr = flt.threshold_otsu(image_gray)
    img = image_gray < thr
    filled = scipy.ndimage.morphology.binary_fill_holes(img)
    label = measure.label(filled)

    for cell in measure.regionprops(label):
        eccentricity = cell.eccentricity
        
    return [contrast[0][0], dissimilarity[0][0], homogeneity[0][0], ASM[0][0], energy[0][0], correlation[0][0], ent, eccentricity]

    
