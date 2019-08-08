#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:34:46 2019

@author: reganlamoureux
"""

import matplotlib.pyplot as plt
import numpy as np
import math

import skimage
import os

from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage.morphology import watershed, disk
from skimage.transform import hough_circle
from skimage.util import img_as_ubyte


set_number = 1
area_dict = {}
all_area = 0
my_image = ''


def threshold_image(picture):
    #newImage = Image.new(picture.mode, picture.size)
    image = np.asanyarray(picture)
    image_threshold = filters.threshold_otsu(image)
    image_array = image < image_threshold
    return image_array

def image_histogram(a_dictionary):
    new_dict = a_dictionary.copy()
    for key in a_dictionary.keys():
        if a_dictionary[key] > 10:
            del new_dict[key]
    plt.bar(new_dict.keys(), new_dict.values())
    plt.ylim(0, 15)
    plt.xlim(0, 25000)
    plt.show()

def watershed_segmentation(image):
    image = img_as_ubyte(image)

    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(1))<10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))
    
    thr = filters.threshold_yen(image)
    bw_img = image < thr
    bw_img = skimage.morphology.erosion(bw_img,
                                         disk(3))
    # close holes, size check
    label_image = skimage.morphology.remove_small_holes(bw_img,
                                                        area_threshold=625)
    label_image = skimage.morphology.remove_small_objects(label_image,
                                                          min_size=100)
    label_image = measure.label(label_image, background=0)

    markers = skimage.morphology.erosion(label_image,
                                         disk(1))
    # process the watershed
    labels = watershed(gradient, label_image)
    counter = 1
    config = '/Users/reganlamoureux/new_watershed_images/watershed_image_set2_{}.png'.format(counter)
    while os.path.exists(config):
        counter += 1
        config = '/Users/reganlamoureux/new_watershed_images/watershed_image_set2_{}.png'.format(counter)
    plt.imsave(config, labels)
    
    return labels


def other_water(image):
    b_tophat = morphology.black_tophat(image, disk(4))
    thresholded_image = threshold_image(b_tophat)

    distance = ndi.distance_transform_edt(thresholded_image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=image)


    markers = ndi.label(local_maxi)[0]

                            
    labels = watershed(thresholded_image, markers)
    label_image = skimage.morphology.remove_small_objects(labels,
                                                              min_size=100)
    
    counter = 1
    config = '/Users/reganlamoureux/other_watershed_images/watershed_image{}.png'.format(counter)
    while os.path.exists(config):
        counter += 1
        config = '/Users/reganlamoureux/new_watershed_images/watershed_image{}.png'.format(counter)
    plt.imsave(config, label_image)



def connected_component_labels(image_array):
    global area_dict
    global all_area
    global set_number
    filled_image = ndi.binary_fill_holes(image_array)
    eroded_img = morphology.erosion(filled_image)
    gaussian_img = filters.gaussian(eroded_img)
    label_image = measure.label(gaussian_img, background=0)
    #thresholded_image = thresholdImage2(label_image)

    my_list = []
    counter = 0
    for cell in measure.regionprops(label_image):
        if cell.area>1000:
            set_number+=1
            all_area = all_area + cell.area
            if cell.area not in area_dict:
                area_dict[cell.area] = 1
            else:
                area_dict[cell.area] += 1
            center_of_mass_y, center_of_mass_x = cell.centroid
            #ax2.scatter(center_of_mass_x, center_of_mass_y, c='white')
            my_list.append((center_of_mass_x, center_of_mass_y))
        counter = counter + 1
    sec_count = 1 
    config = '/Users/reganlamoureux/connected_images/labeled_image_set_{}.png'.format(sec_count)
    while os.path.exists(config):
        sec_count += 1
        config = '/Users/reganlamoureux/connected_images/labeled_image_set_{}.png'.format(sec_count)
        
    plt.imsave(config, label_image)

        
    return label_image


def crop_image(image):
    counter = 0
    reg_props = measure.regionprops(image)
    my_objects = ndi.find_objects(image)
    for cell, cur_slice in zip(reg_props, my_objects):
        if cell.area>1000:
            config = '/Users/reganlamoureux/cropped_originals/cropped_original_image_{}.png'.format(counter)
            while os.path.exists(config):
                counter += 1
                config = '/Users/reganlamoureux/cropped_originals/cropped_original_image_{}.png'.format(counter)
            cropped_image = my_image[cur_slice]
            plt.imsave(config, cropped_image) 
    
def crop_image2(image):
    global set_number
    global my_image
    counter = 0
    reg_props = measure.regionprops(image)
    my_objects = ndi.find_objects(image)
    for cell, cur_slice in zip(reg_props, my_objects):
        if cell.area>1000:
            config = '/Users/reganlamoureux/watershed_cropped/watershed_cropped_image_set_{}_image_{}.png'.format(set_number, counter)
            while os.path.exists(config):
                counter += 1
            cropped_image = image[cur_slice]
            plt.imsave(config, cropped_image)
            counter += 1
    set_number += 1 
    
def crop_image3(image):
    counter = 1
    reg_props = measure.regionprops(image)
    my_objects = ndi.find_objects(image)
    for cell, cur_slice in zip(reg_props, my_objects):
        if cell.area>1000:
            minr, minc, maxr, maxc = cell.bbox
            #rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  #fill=False, edgecolor='red', linewidth=2)
                
            x, y, w, h = minr-5, minc-5, maxr+5, maxc+5 # make the box a little bigger
            if y<0:
                y = 0
            if x<0:
                x = 0
            if h > 696:
                h = 696
            if w>520:
                w=520
            config = '/Users/reganlamoureux/watershed_cropped/watershed_cropped_original_image_{}.png'.format(counter)
            while os.path.exists(config):
                counter += 1
                config = '/Users/reganlamoureux/watershed_cropped/watershed_cropped_original_image_{}.png'.format(counter)
            #cropped_image = my_image[cur_slice]
            cropped_image = my_image[x:w, y:h]
            plt.imsave(config, cropped_image)
            counter = counter + 1
            
            
def hough_transform(image):
    #ubyte_image = img_as_ubyte(image)
    region = measure.regionprops(image)
    i = 0
    for cell in region:
        if cell.area>100:
            radius = int(math.sqrt(((cell).coords[0][1]-(cell).centroid[1])**2
                                   +((cell).coords[0][0]-(cell).centroid[0])**2))
            hough_res = hough_circle(image, radius)
            circy, circx = circle_perimeter(int(cell.centroid[0]), int(cell.centroid[1]), radius)
            try:
                image[circy, circx] = 255
            except:
                pass
        i+=1
        

def run_processing(image):
    global my_image
    my_image = skimage.io.imread('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution/'+ image)
    #thres_img = threshold_image(my_image)
    #label_img = connected_component_labels(thres_img)
    #crop_image3(label_img)
    
    #water_img = watershed_segmentation(my_image)
    #sec_label_img = connected_component_labels(water_img)
    #crop_image3(water_img)
    other_water(my_image)
    
def edit_images(my_path):
    folder = os.fsdecode(my_path)
    counter = 0
    list_dir = []
    #return os.listdir(folder)
    #for image in os.listdir(folder):
    for image in os.listdir(folder):
        #print(str(counter) + '. ' + image)
        if image=='.DS_Store':
            pass
        else:
            #run_processing(image)
            list_dir.append(image)
            counter+=1
    return list_dir
        
list_dir = edit_images('/Users/reganlamoureux/UVM-2019-05-30-15-45-12 Malaria 2x dilution')
'''image_histogram(area_dict)
print(set_number)
print('----------------')
print(all_area)
print('----------------')
print(all_area/set_number)
print('----------------')
print(max(area_dict.values()))
#imageHistogram(area_dict)

#print(list_dir)
'''
def joblib_loop():
    Parallel(n_jobs=4)(delayed(run_processing)(i) for i in list_dir)
joblib_loop()



