# -*- coding: utf-8 -*-
"""Dataset_Generation.ipynb

**IMPORT LIBRARIES**
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import drive
from google.colab import files
drive.mount('/content/drive')

"""**PROCESSING IMAGES FOR OTTER DATASET**"""

IMAGE_SIZE = 256
n_img = 50
path_read = "/content/drive/MyDrive/...PATH.../RAW_IMG/img ("
path_write = "/content/drive/MyDrive/...PATH.../Processed_IMG/img_p ("
#In case the images need to be cropped, a region of interest is defined Considering 2HD resolution (1242x4416).
y, h = 0, int(1242*0.73)
x, w = 0, int(4416/2)
for i in range (n_img):
  #Read Image.
  path_img = path_read + str(i+1) + ").png"
  img = cv2.imread(path_img)
  #Crop Image.
  crop_img = img[y:y+h, x:x+w]
  #Save Image.
  path_img = path_write + str(i) + ").png"
  cv2.imwrite(path_img, crop_img)

"""**PROCESSING MASKS FROM MaSTr1325 DATASET - TO BINARY CLASSES**"""

#Define path of original output mask. 
path_msk = "/content/drive/MyDrive/...PATH.../Process_IMG/MSK/"
#Define path of binary output mask. 
path_msk_binary = "/content/drive/MyDrive/...PATH.../Process_IMG/MSK/"
n_input_img = 1000
for i in range(n_input_img):
    #Read output mask.
    path_msk_index = path_msk + "mask_" + str(i+1) + ".png"
    mask = cv2.imread(path_msk_index)
    #Obstacles and environment = 0 (value zero)
    mask[mask > 0] = 1
    path_msk_binary_index = path_msk_binary + "mask (" + str(i+1) + ").png"
    cv2.imwrite(path_msk_binary_index, mask)

"""**PROCESSING MASKS FROM MATLAB LABELING - TO CORRECT BINARY CLASSES**"""

#Define path of original MATLAB output masks. 
path_msk = "/content/drive/MyDrive/...PATH.../Process_IMG/MSK/"
#Define path of modified output mask. 
path_msk_binary = "/content/drive/MyDrive/...PATH.../Process_IMG/MSK/Binary_MSK/"
n_input_img = 1000
for i in range(n_input_img):
    path_msk_index = path_msk + "Label_" + str(i+1) + ".png"
    mask = cv2.imread(path_msk_index, cv2.IMREAD_GRAYSCALE)
    #After MATLAB labeling, obstacles and environment class is 1 and non obstacle is 0, so with this procedure the classes are flipped to have the same as the original dataset.
    mask[mask == 0] = 2
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    path_msk_binary_index = path_msk_binary + "mask_" + str(i+1) + ".png"
    cv2.imwrite(path_msk_binary_index, mask)