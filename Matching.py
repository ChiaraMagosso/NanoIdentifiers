## 
## Copyright (C) 2024 by
## Stefano Carignano & Chiara Magosso
## 
## This work is licensed under a  
## 
## Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
## ( http://creativecommons.org/licenses/by-nc-sa/4.0/ )
## 
## Please contact stefano.carignano@bsc.es & chiara.magosso@polito.it for information.
##
## This code is associated with the following work : https://doi.org/10.21203/rs.3.rs-4170364/v1
##

# Reference : https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

""" 
The input should be two set of already binarized images contained in the folders
ref_folder and cfr_folder, defined a few lines below (change those accordingly)

The algorithm goes as follows: for each image pair (img1, img2) we want to compare,
we use the SIFT algorithm to identify keypoints in both images,
then we run a matcher to find the matching keypoints in both images,
then using those we build a transformation matrix to find the best mapping of one image on 
top of the other using the cv.findHomography() [should account for translations, rotations
and deformations],
we now map one image on top of the other, and check the overlap between the two, 
using the function mse() (which uses np.logical_xor() to do the overlap).

It returns the resulting correlation matrices as txt: 'corr_mat_overlap.txt'
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np 
import os
from scipy.spatial import distance
import sys

MIN_MATCH_COUNT = 9

def check_binary_image(img, range_type):
    if range_type == '0-255':
        condition = np.logical_and(img != 0, img != 255)
    elif range_type == '0-1':
        condition = np.logical_and(img != 0, img != 1)
    else:
        raise ValueError("Invalid range type. Please choose '0-255' or '0-1'.")
    if np.any(condition):
        print('Attenzione: immagine non binaria!')
        sys.exit()

def mse(img1,name_img1, img2,name_img2):
    if img1.shape != img2.shape: 
        print('Diverse dimensioni')
        sys.exit()
    logica = np.logical_xor(img2, img1)
    diff =  logica.astype(int)
    h, w = diff.shape
    HD_XOR = np.count_nonzero(diff)/(h*w)
    
    check_binary_image(diff, range_type='0-1')

    plt.imshow(diff, cmap='Purples_r', aspect='equal')

    plt.tick_params(left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)
    plt.savefig(f'{output}overlap_{name_img1}_{name_img2}.png',  dpi=500)
    plt.clf()

    err = distance.hamming(list(np.concatenate(img2.astype(bool).astype(int)).flat), list(np.concatenate(img1.astype(bool).astype(int)).flat))
    
    if HD_XOR == err:
        pass
    else: 
        print('HD non corrisponde a XOR') 
        sys.exit()
    mse = err  
    return mse

def calc_mse(img1,name_img1, img2, name_img2):
    cv.setRNGSeed(666)
    h_base, w_base = img1.shape
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 8)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img2.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        warped_img1 = cv.warpPerspective(img1, M, (w, h))

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        warped_img1 = img1

    ret,warped_img1_thresh = cv.threshold(warped_img1,127,255,cv.THRESH_BINARY)
    check_binary_image(warped_img1_thresh, range_type='0-255')
    plt.imshow(warped_img1_thresh, cmap='Reds_r', aspect='equal')
    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.savefig(f'{output}warped_binary_{name_img1}_with_{name_img2}.png', dpi=500)  
    plt.clf()

    check_binary_image(img2, range_type='0-255')

    plt.imshow(img2, cmap='Blues_r', aspect='equal')
    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.savefig(f'{output}not_warped_{name_img1}_with_{name_img2}.png', dpi=500)
    plt.clf()

    mserr = mse(warped_img1,name_img1, img2, name_img2)
    draw_params = dict(
    matchesThickness = 8,
    matchesMask = matchesMask, 
    flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, aspect='equal')
    plt.tick_params(left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)
    plt.savefig(f'{output}matches_{name_img1}_{name_img2}.png', dpi=500)
    plt.clf()

    return mserr

ref_folder = 'Matching_data/ref/' 
cfr_folder = 'Matching_data/crf/'
output='Matching_data/comparison/'

set_1 = os.listdir(ref_folder)
set_2 = os.listdir(cfr_folder)

for el in set_1:
    if not el.endswith(".tif"):
        set_1.remove(el)

for el in set_2:
    if not el.endswith(".tif"):
        set_2.remove(el)

set_1 = sorted(set_1) 
set_2 = sorted(set_2)

print(set_1, len(set_1))
print(set_2, len(set_2))

matrix_matches = np.zeros((len(set_1), len(set_2)))

for ii, ref_image in enumerate(set_1):
    if ref_image.endswith(".tif"):
        img1 = cv.imread(ref_folder + ref_image, cv.IMREAD_GRAYSCALE) # queryIma
        check_binary_image(img1,range_type='0-255')

    for jj, cfr_image in enumerate(set_2):
        if cfr_image.endswith(".tif"):
            img2 = cv.imread(cfr_folder + cfr_image, cv.IMREAD_GRAYSCALE) # queryIma
            check_binary_image(img2, range_type='0-255')
            n_good_matches = calc_mse(img1,ref_image, img2, cfr_image)
            print('Compering : ',ref_image,'with : ', cfr_image, ', HD : ',n_good_matches)
            matrix_matches[ii,jj] = n_good_matches

for ii in range(matrix_matches.shape[0]):
   jbest = np.argmin(matrix_matches[ii,:])
   #print(jbest, matrix_matches[ii,jbest])

np.savetxt(f'{output}corr_mat_overlap.txt', matrix_matches)
plt.imshow(matrix_matches)
plt.colorbar()
plt.savefig(f'{output}match_matrix.png')
plt.clf()