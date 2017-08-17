"""
This file loads in and uses a pre-trained model for testing purposes

Author : Ali Shelton
"""

import numpy as np
import cv2
import keras
import pandas as pd
import os
from keras.optimizers import Adam

from fetch_data import (
	fetch_data,
	format_inputs,
	combo_generator,
	demo_generator,
	split_data
)

from keras.preprocessing.image import ImageDataGenerator


OUTPUT_PATH = 'C:\\Users\\alishelton\\Documents\\analysis_limited_hip'
INPUT_PATH = 'C:\\Users\\alishelton\\Documents\\numpy_arrs\\limited_hip'
MODEL_PATH = 'C:\\Users\\alishelton\\Documents\\models\\res_model_limited_hip.h5'


fetched = fetch_data(INPUT_PATH, combined=False, scan_type='Hip')
train, val, test = format_inputs(fetched)

x_test, y_test = test
print(x_test.shape)
x_train, y_train = train
print(x_train.shape)

# val_gen = combo_generator(x_vals[0], x_vals[1], y_val)

# Multi image generator 
#test_gen = combo_generator(x_test[0], x_test[1], y_test, include_y=False)

# Demographic info
"""
real_age = np.load('C:\\Users\\alishelton\\Documents\\numpy_arrs\\demographics\\real_age.npy')
sex = np.load('C:\\Users\\alishelton\\Documents\\numpy_arrs\\demographics\\sex.npy')
age_split, sex_split = split_data([real_age, sex]) 
age_val, age_test, age_train = age_split
sex_val, sex_test, sex_train = sex_split
test_gen = demo_generator(x_test, age_test, sex_test, y_test)
test_gen1 = demo_generator(x_test, age_test, sex_test, y_test, include_y=False)

"""


# Single image generator
test_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
test_gen.fit(x_test)


model = keras.models.load_model(MODEL_PATH)
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mae'])


test_pred = model.predict_generator(test_gen.flow(x_test, shuffle=False), steps=x_test.shape[0] / 32)
print(model.evaluate_generator(test_gen.flow(x_test, y_test, shuffle=False), steps=x_test.shape[0] / 32))
np.savetxt(os.path.join(OUTPUT_PATH, 'y_pred.txt'), test_pred)
np.savetxt(os.path.join(OUTPUT_PATH, 'y_test.txt'), y_test)


"""
val_pred = model.predict(x_vals, batch_size=32, verbose=1)
np.savetxt('C:\\Users\\alishelton\\Documents\\analysis_distributed_large_paired\\val_predicts.txt', val_pred)
np.savetxt('C:\\Users\\alishelton\\Documents\\analysis_distributed_large_paired\\y_val.txt', y_val)
"""