# -*- coding: utf-8 -*-
"""Video_Production.ipynb

**IMPORT LIBRARIES**
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import drive
from google.colab import files
import tensorflow as tf
drive.mount('/content/drive')

"""**UTILITY FUNCTIONS**"""

def f1_score_coef(y_true, y_pred):
    #Constant to avoid division by zero.
    smooth = 1e-7
    #Flatten real target data into 1D array.
    y_true = tf.keras.layers.Flatten()(y_true)
    #Flatten prediction target data into 1D array.
    y_pred = tf.keras.layers.Flatten()(y_pred)
    #Apply F1_Score formula:
    intersection = tf.reduce_sum(y_true * y_pred)
    f1_score_coef = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return f1_score_coef

def f1_score_loss(y_true, y_pred):
    #Calculate F1_Score Loss from coefficient.
    return 1.0 - f1_score_coef(y_true, y_pred)

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

"""**VIDEO GENERATION**"""

IMAGE_SIZE = 256
test_model = tf.keras.models.load_model('/content/drive/MyDrive/...PATH.../Models/otternetvx.h5', custom_objects={'dice_loss':f1_score_loss, 'dice_coef':f1_score_coef})
video_in = "/content/drive/MyDrive/...PATH.../Video/IN/record_X.avi"
video_out = "/content/drive/MyDrive/...PATH.../Video/OUT/record_X"
frame_path =  "/content/drive/MyDrive/...PATH.../Video/temp_frame.png"

#Setup video saving properties.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#Output is prepared to receive input image rotated 90 degrees.
name = str(video_out) + '_output.avi'
video_out = cv2.VideoWriter(name, fourcc, 15, (695,348))

video_frames = cv2.VideoCapture(video_in)
video_length = int(video_frames.get(cv2.CAP_PROP_FRAME_COUNT))

for frame in range(video_length): 
  y, h = 0, int(1242*0.73)
  x, w = 0, int(4416/2)
  ret, img = video_frames.read()
  crop_img = img[y:y+h, x:x+w]
  img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
  img = img/255.0

  y_pred = test_model.predict(np.expand_dims(img, axis=0))[0]
  y_pred[y_pred > 0.5] = 255
  h, w, _ = img.shape
  white_line = np.ones((h, 10, 3))
  all_images = [img, white_line, mask_parse(y_pred)]

  image = np.concatenate(all_images, axis=1)
  fig = plt.figure(figsize=(12, 12))
  a = fig.add_subplot(1, 1, 1)
  imgplot = plt.imshow(image)
  plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
  plt.clf()
  frame_img = cv2.imread(frame_path)
  video_out.write(frame_img)

video_out.release()