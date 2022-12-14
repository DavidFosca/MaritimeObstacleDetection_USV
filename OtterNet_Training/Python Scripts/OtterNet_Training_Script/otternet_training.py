# -*- coding: utf-8 -*-
"""OtterNet_Training

**IMPORT LIBRARIES**
"""

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from google.colab.patches import cv2_imshow
from google.colab import drive
from google.colab import files
drive.mount('/content/drive')

"""**UTILITY FUNCTIONS AND UNET NETWORK DEFINITION**"""

IMG_SIZE = 256
path_img = "/content/drive/MyDrive/...PATH.../MaSTr1325_images_512x384/"
path_target = "/content/drive/MyDrive/...PATH.../MaSTr1325_masks_512x384_binary/"
epochs = 30
batch = 32
learning_rate = 1e-4

def load_img(path_img, path_target,split):
    #Obtain all the file paths for the input images and output targets. 
    images = sorted(glob(os.path.join(path_img, "*")))
    target = sorted(glob(os.path.join(path_target, "*")))
    #Randomly select 10% of the entire Dataset as Validation data.  
    train_x, valid_x = train_test_split(images, test_size=int(split * len(images)), random_state=42)
    train_y, valid_y = train_test_split(target, test_size=int(split * len(images)), random_state=42)
    #Randomly select 10% of the entire Dataset as Testing data.
    train_x, test_x = train_test_split(train_x, test_size=int(split * len(images)), random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=int(split * len(images)), random_state=42)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    #Get path from file image.
    path = path.decode()
    #Read image from path using OpenCV.
    img = cv2.imread(path)
    #Resize image to 255x255.
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    #Normalize image.
    img = img/255.0
    return img

def read_target(path):
    #Get path from file target.
    path = path.decode()
    #Read image from path as greyscale using OpenCV.
    msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #Resize target to 255x255.
    msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE))
    #Define target at floating point.
    msk = msk/1.0
    #Add one dimension to image array.
    msk = np.expand_dims(msk, axis=-1) #for grayscale
    return msk

def call_convert(img, msk):
    def _convert(img, msk):
        #Call data processing functions.
        img = read_image(img)
        msk = read_target(msk)
        return img, msk
    #Define image and target mask formats.
    img, msk = tf.numpy_function(_convert, [img, msk], [tf.float64, tf.float64])
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    msk.set_shape([IMG_SIZE, IMG_SIZE, 1])

    return img, msk

trainAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255),
	preprocessing.RandomFlip("horizontal"),
	preprocessing.RandomZoom(height_factor=(-0.05, -0.15), width_factor=(-0.05, -0.15)),
	preprocessing.RandomRotation(0.2)])

def parse_dataset(img, msk, BATCH):
    #Process input image and target mask data through map function.
    data_set = tf.data.Dataset.from_tensor_slices((img, msk))
    data_set = data_set.map(call_convert)
    data_set = (data_set.shuffle(BATCH*100).batch(BATCH).map(lambda i, j: (trainAug(i), j), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
    data_set = data_set.repeat()
    return data_set
  
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_img(path_img, path_target,split=0.1)

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

def OtterNet():
    #Define input layer with size and name. 
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    #Load MobileNetV2:
    #input layer is assigned to input of model.
    #weights are pre-training on ImageNet
    #do not include a fully-connected layer at the top.
    #alpha > 1 will proportionally increase the number of filters in each layer. 
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=1.3)
    #Get MobileNetV2 specific layer output.
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    #Get MobileNetV2 specific layer output as skip connection.
    x_skip = encoder.get_layer("block_6_expand_relu").output
    x = UpSampling2D((2, 2))(encoder_output)
    x = Concatenate()([x, x_skip])
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #Get MobileNetV2 specific layer output as skip connection.
    x_skip = encoder.get_layer("block_3_expand_relu").output
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_skip])
    x = Conv2D(48, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(48, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #Get MobileNetV2 specific layer output as skip connection.
    x_skip = encoder.get_layer("block_1_expand_relu").output
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_skip])
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #Get MobileNetV2 specific layer output as skip connection.
    x_skip = encoder.get_layer("input_image").output
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_skip])
    x = Conv2D(16, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(16, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)   
    #Last layer.
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    OtterNet = Model(inputs, x)
    return OtterNet

"""**TRAINING**"""

#Call OtterNet model.
model = OtterNet()
model.summary()
#Compile model with: loss, optimizer with learning rate, and f1-score, recall and precision as metrics.
model.compile(loss=f1_score_loss, optimizer=tf.keras.optimizers.Nadam(learning_rate), metrics=[f1_score_coef, Recall(), Precision()])
#Callbacks definition for early stopping and reducing learning rate.
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4), EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)]

#Process and load train and validation dataset using the utility function.
train_dataset = parse_dataset(train_x, train_y, batch=batch)
valid_dataset = parse_dataset(valid_x, valid_y, batch=batch)

#Number of train and validation steps are needed to point out how many batches (integer number) will be executed during an epoch.
train_steps = len(train_x)//batch
if len(train_x) % batch != 0: train_steps += 1
valid_steps = len(valid_x)//batch
if len(valid_x) % batch != 0: valid_steps += 1
#The model starts training with the selected train and validation dataset, number of epochs, steps to run per epoch for training and validation, and the two callback functions. 
history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=callbacks)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

model.save("/content/drive/MyDrive/...PATH.../otternetvx.h5")

"""**TESTING**"""

#Load OtterNet trained model.
test_model = tf.keras.models.load_model("/content/drive/MyDrive/...PATH.../otternetvx.h5", custom_objects={'dice_loss':f1_score_loss, 'dice_coef':f1_score_coef})
#Process and load test dataset using the utility function.
test_dataset = parse_dataset(test_x, test_y, batch=batch)
#Number of test steps are needed to point out how many batches (integer number) will be executed during an epoch.
test_steps = (len(test_x)//batch)
if len(test_x) % batch != 0: test_steps += 1
#Evaluate model will give back the metric scores.
test_model.evaluate(test_dataset, steps=test_steps)