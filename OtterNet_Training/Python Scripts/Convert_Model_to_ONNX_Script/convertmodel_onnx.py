# -*- coding: utf-8 -*-
"""ConvertModel_ONNX.ipynb

**IMPORT LIBRARIES**
"""

!pip install tf2onnx
!pip install onnxruntime
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as rt
import numpy as np
import cv2
from google.colab import drive
from google.colab import files
import matplotlib.pyplot as plt
from datetime import datetime
from onnxruntime.quantization import quantize_dynamic, QuantType
drive.mount('/content/drive')

"""**DEFINE FUNCTIONS**"""

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

def read_image(path):
    #Read image from path using OpenCV.
    img = cv2.imread(path)
    #Resize image to 256x256x3.
    img = cv2.resize(img,(256,256))
    #Normalize image.
    img = img/255.0
    return img

def read_target(path):
    #Read image from path as greyscale using OpenCV.
    msk = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    #Resize target to 256x256.
    msk = cv2.resize(msk,(256,256))
    #Define target at floating point.
    msk = msk/1.0
    #Add one dimension to image array.
    msk = np.expand_dims(msk,axis=-1)
    return msk

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

"""**CONVERT KERAS MODEL TO ONNX FORMAT**"""

#Read Keras model from path.
model = tf.keras.models.load_model('/content/drive/MyDrive/...PATH.../Models/otternetvx.h5', custom_objects={'dice_loss':f1_score_loss, 'dice_coef':f1_score_coef})
#Define input tensor type and name.
format = (tf.TensorSpec((None, 256, 256, 3), tf.float32, name="input_image"),)
#Convert Keras model to ONNX. Optset depends on the Jetpack version, in this case it has to be 12 because 13 is not supported.
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature = format, opset=12)
#Define path to save ONNX model.
onnx_path = "/content/drive/MyDrive/...PATH.../Models/otternetvx.onnx"
#Save ONNX model.
onnx.save(onnx_model, onnx_path)

"""**TEST ONNX MODEL**"""

#Load ONNX model.
onnx_path = "/content/drive/MyDrive/...PATH.../Models/otternetvx.onnx"
onnx_model = onnx.load(onnx_path)
#Read input and output graph names from ONNX model. This is later needed to run inferece.
output_names = [n.name for n in onnx_model.graph.output]
input_names = [n.name for n in onnx_model.graph.input]
#Crete ONNX Inference Session.
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(onnx_path, providers=providers)

#Run Inference and Test on Dataset.
path_img_test = "/content/drive/MyDrive/...PATH.../Process_IMG/IMG/"
n_test = 50
start=datetime.now()
for i in range(n_test):
    path_x = path_img_test + "img_" + str(i+1) + ".png"
    x = read_image(path_x)
    #Run Inference.
    onnx_pred = m.run(output_names, {"input_image": np.expand_dims(x, axis=0).astype(np.float32)})[0]
    #Step needed to plot the predicted output mask (black and white)
    onnx_pred[onnx_pred > 0.5] = 255
    h, w, _ = x.shape
    white_line = np.ones((h, 10, 3))
    #Concatenate Input Image, Output Target, and Prediction.
    all_images = [x, white_line, mask_parse(onnx_pred)]
    image = np.concatenate(all_images, axis=1)
    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image)
print(datetime.now()-start)

"""**POST-TRAINING QUANTIZATION AND TESTING**"""

#Define path of original ONNX 32-bit fp model.
model_fp32 = "/content/drive/MyDrive/...PATH.../Models/otternetvx.onnx"
#Define quantized model saving path.
model_quant = "/content/drive/MyDrive/...PATH.../Models/otternetvx.quant.onnx"
#Model Quantization to INT8.
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

#Load Quantized Model and Run inference to test dataset on new model.
onnx_model = onnx.load(model_quant)
output_names = [n.name for n in onnx_model.graph.output]
input_names = [n.name for n in onnx_model.graph.input]
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(model_quant, providers=providers)
path_img_test = "/content/drive/MyDrive/...PATH.../Process_IMG/IMG/"
n_test = 50
start=datetime.now()
for i in range(n_test):
    path_x = path_img_test + "img_" + str(i+1) + ".png"
    x = read_image(path_x)
    #Run Inference.
    onnx_pred = m.run(output_names, {"input_image": np.expand_dims(x, axis=0).astype(np.float32)})[0]
    #Step needed to plot the predicted output mask (black and white)
    onnx_pred[onnx_pred > 0.5] = 255
    h, w, _ = x.shape
    white_line = np.ones((h, 10, 3))
    #Concatenate Input Image, Output Target, and Prediction.
    all_images = [x, white_line, mask_parse(onnx_pred)]
    image = np.concatenate(all_images, axis=1)
    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image)
print(datetime.now()-start)