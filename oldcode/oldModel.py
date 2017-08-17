import numpy as np
import cv2
import keras
import pandas as pd
import os
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import RMSprop, Adam
from keras.applications.resnet50 import ResNet50


########## Data Generation and Pre-processing ##########
df = pd.read_excel('C:\\Users\\alishelton\\Documents\\all_patient_info.xlsx')
files = df['pfiles'].values 
ids = df['IDNO'].values
files = [pfile[0:8] + '_' + str(id_num) + pfile[8:] for pfile, id_num in zip(files, ids)]
types = df['types'].values
sexes = df['sexes'].values
eths = df['eths'].values
bone_age = df['YEAR'].values + (df['MONTH'].values / 12)
patInfo = dict(zip(files, zip(types, sexes, bone_age)))

# Training data x values 
"""
Possible pre-processing techniques:
1. per-channel normalization <-- seems the most viable for preventing saturated neurons
2. per-channel mean subtraction
"""
x_train_pre = []
y_train_pre = []

# This method of forming data ensures the image and output bone age match up
for key in patInfo:
	if patInfo[key][0] == 'RightHip':
		img = np.array(cv2.imread(os.path.join('C:\\Users\\alishelton\\Documents\\Testing', key)  + '.jpg', cv2.IMREAD_GRAYSCALE))
		img = np.reshape(img, (224, 224, 1))
		x_train_pre.append(img)
		y_train_pre.append(patInfo[key][2])

x_train, y_train = np.array(x_train_pre), np.array(y_train_pre)

# Per Channel Normalization
x_train_min = x_train.min(axis=(1, 2), keepdims=True)
x_train_max = x_train.max(axis=(1, 2), keepdims=True)
x_train = (x_train - x_train_min)/(x_train_max-x_train_min)

x_test, x_train = x_train[5000:], x_train[:5000]
y_test, y_train = y_train[5000:], y_train[:5000]

########## Model Building ##########

### Hyperparameters
drop_prob1 = 0.25
drop_prob2 = 0.5
conv_filters1 = 32
conv_filters2 = 64
filter_size = (3,3)
pool_size = (2,2)

### Layers

#Convolutional layers
conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last', input_shape=(224, 224, 1))
conv2 = Conv2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv3 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv4 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv4 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv5 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv6 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv7 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv8 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv9 = Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv10 = Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv11 = Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')
conv12 = Conv2D(128, (3, 3), padding='same', activation='relu', data_format='channels_last')


#Pooling layers
pool1 = MaxPooling2D(pool_size=(2,2), padding='same')
pool2 = MaxPooling2D(pool_size=(2,2), padding='same')
pool3 = MaxPooling2D(pool_size=(2,2), padding='same')
pool4 = MaxPooling2D(pool_size=(2,2), padding='same')
pool5 = MaxPooling2D(pool_size=(3,3), padding='same')
pool6 = MaxPooling2D(pool_size=(3,3), padding='same')

#Fully-connected and flattening layers
flatten = Flatten()
dense1 = Dense(512, activation='relu')
dense2 = Dense(1, activation='linear')

#Dropout layers to prevent overfitting on data, these layers can potentially be added later down the road 
drop1 = Dropout(0.25) 
drop2 = Dropout(0.25)
drop3 = Dropout(0.5)

### Model optimizer

opt = RMSprop()

### Sequential Net

model = Sequential()

#Build the model by adding layers
model.add(conv1)
model.add(conv2)
model.add(pool1)
model.add(drop1)
model.add(conv3)
model.add(conv4)
model.add(pool2)
model.add(conv5)
model.add(conv6)
model.add(pool3)
model.add(conv7)
model.add(conv8)
model.add(pool4)
model.add(conv9)
model.add(conv10)
model.add(pool5)
model.add(conv11)
model.add(conv12)
model.add(pool6)
model.add(drop2)
model.add(flatten)
model.add(dense1)
model.add(drop3)
model.add(dense2)

# Compile the model
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

########## Training and Analyzing Output ##########

#Train the model on the input dataset and get analytics
hist = model.fit(x_train, y_train, epochs=50, batch_size=64)
print(hist.history)

#model2 = ResNet50(include_top=True, weights=None, pooling=max, classes=12)
#model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#model2.fit(x_train, y_train, epochs=2, batch_size=32)

print(model.evaluate(x_test, y_test))

model.save('C:\\Users\\alishelton\\Documents\\cnn_model.h5')

























########### OUT OF DATE #############
def ageFromDates(s_date, b_date):
	age = s_date.year - b_date.year
	if b_date.month < s_date.month or b_date.month == s_date.month and b_date.day < s_date.day:
		return age
	return age - 1
