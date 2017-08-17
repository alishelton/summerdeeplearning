"""
This file trains the model and saves relevant loss and error information during the training process

LR Reduction Source : https://github.com/fchollet/keras/issues/898

Author : Ali Shelton
"""

import numpy as np
import cv2
import keras
import pandas as pd
import os
import resnet
import inception_v4

from keras.optimizers import (
	RMSprop, 
	Adam, 
	Nadam, 
	SGD
)

from fetch_data import (
	fetch_data,
	format_inputs,
	combo_generator,
	split_data,
	demo_generator
)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

########## VARS ############
INPUT_PATH = 'C:\\Users\\alishelton\\Documents\\numpy_arrs\\limited_hip'
MODEL_PATH = 'C:\\Users\\alishelton\\Documents\\models\\res_model_limited_hip.h5'
OUTPUT_PATH = 'C:\\Users\\alishelton\\Documents\\analysis_limited_hip'
include_val = False


########## PARAMS #################
NUM_CLASSES = 1
DROPOUT_PROB = 0.5
INPUT_SHAPE = (1, 130, 250)


########## Data Generation and Pre-processing ##########
fetched = fetch_data(INPUT_PATH, combined=False, scan_type='Hip')
train, val, test = format_inputs(fetched)

x_train, y_train = train
x_val, y_val = val


# demo data
"""
real_age = np.load('C:\\Users\\alishelton\\Documents\\numpy_arrs\\demographics\\real_age.npy')
sex = np.load('C:\\Users\\alishelton\\Documents\\numpy_arrs\\demographics\\sex.npy')
age_split, sex_split = split_data([real_age, sex]) 
age_val, age_test, age_train = age_split
sex_val, sex_test, sex_train = sex_split

train_gen = demo_generator(x_train, age_train, sex_train, y_train)
val_gen = demo_generator(x_val, age_val, sex_val, y_val)
"""

########## Preprocessing ##########


# Paired preprocessing
#train_gen = combo_generator(x_train[0], x_train[1], y_train)
#val_gen = combo_generator(x_val[0], x_val[1], y_val)


# Single preprocessing
train_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
train_gen.fit(x_train)

if include_val:
	val_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
	val_gen.fit(x_val)


########## Model Building ##########
model = resnet.ResnetBuilder.build_resnet_34(INPUT_SHAPE, NUM_CLASSES)
# model = inception_v4.inception_v4(NUM_CLASSES, DROPOUT_PROB, weights=None, include_top=True)
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mae'])


########## Callbacks ##################
class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs = {}):
		self.losses = []

	def on_batch_end(self, batch, logs = {}):
		self.losses.append(logs.get('loss'))

########## Training and Analyzing Output ##########

#Train the model on the input dataset and get analytics
histories = LossHistory()
lr_reducer = ReduceLROnPlateau(factor=0.5, patience=4, cooldown=4) 

hist = model.fit_generator(train_gen.flow(x_train, y_train, shuffle=False), steps_per_epoch=x_train.shape[0] / 32, \
	epochs=200, callbacks=[histories, lr_reducer])

model.save(MODEL_PATH)
np.savetxt(os.path.join(OUTPUT_PATH, 'epochlosses.txt'), hist.history['loss'])
np.savetxt(os.path.join(OUTPUT_PATH, 'epocherror.txt'), hist.history['mean_absolute_error'])
np.savetxt(os.path.join(OUTPUT_PATH, 'batchlosses.txt'), histories.losses)

if include_val:
	np.savetxt(os.path.join(OUTPUT_PATH, 'valerror.txt'), hist.history['val_mean_absolute_error'])
	np.savetxt(os.path.join(OUTPUT_PATH, 'vallosses.txt'), hist.history['val_loss'])

