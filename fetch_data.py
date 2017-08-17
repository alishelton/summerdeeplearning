"""
This file contains the main function fetch_data, which grabs the required scans, 
formats them as numpy arrays, pairs them with their bone age output, and splits 
the data into training, cross validation, and testing sets. 

Other functions include those for pairing scans together, and shuffling the data.

Author : Ali Shelton
"""

import numpy as np
import cv2
import pandas as pd
import os
import csv
from itertools import zip_longest
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter

"""
Loads the required scans and labels from the filepath and splits them into training, cross-val, and
testing sets

load_base_path : Filepath to look for the numpy arrays in
combined : True if archecture takes in 2 scans at once, False if only 1 scan
scan_type : String for the scan type, either Hip or Spine, only used if combined is False
record : Whether or not to record the split
"""
def fetch_data(load_base_path, combined=False, scan_type=None):
	if not combined:
		x, y = np.load(os.path.join(load_base_path,'x_'+ scan_type.lower() +'_single.npy')), \
		np.load(os.path.join(load_base_path, 'y_' + scan_type.lower() + '_single.npy'))
		x_train = [x, y]
	else:
		x_sp, x_hip, y = np.load(os.path.join(load_base_path,'x_spine_paired.npy')), \
		np.load(os.path.join(load_base_path,'x_hip_paired.npy')), np.load(os.path.join(load_base_path,'y_paired.npy'))
		x_train = [x_sp, x_hip, y]
	
	split = split_data(x_train)

	return split[:-1], split[-1]


"""
Formats the input and output data for a single scan type and saves them as numpy arrs*

*NOTE : Saved as numpy arrs instead of pulled in at run-time for speed

scanType : The type of scan to be used, either Hip or Spine
info_filepath : Filepath to look for the dataframe containing p file info
patInfo : A dictionary with image name keys and target info as values
scan_filepath: Filepath to look for the scans in
save_path : Path to save the numpy arrs to
limit : Max # of values to allow in a bone age category, to prevent poor distribution of data
"""
def singleScanToData(scan_type, info_filepath, scan_filepath, save_path, limit):
	merged = pd.read_excel(info_filepath)

	x, y = [], []
	image_files, bone_ages = merged['Name'] + merged['P_' + scan_type], merged['BA']

	for i, bone_age in enumerate(bone_ages):
		if pd.isnull(bone_age) or pd.isnull(image_files[i]):
			continue

		img = cv2.imread(os.path.join(scan_filepath, image_files[i])  + '.jpg', cv2.IMREAD_GRAYSCALE)

		if img is None or not len(img.shape):
			continue

		img = np.reshape(img, (img.shape[0], img.shape[1], 1))

		x.append(img)
		y.append(bone_age)
		print(i)

	x, y = np.array(x, copy=False), np.array(y, copy=False)
	np.save(os.path.join(save_path, 'x_' + scan_type.lower() + '_single'), x)
	np.save(os.path.join(save_path, 'y_' + scan_type.lower() + '_single'), y)



"""
Takes in the csv file of pairs and saves them as input and output numpy arrs for the model

info_filepath : Filepath to look for the dataframe containing p file info
scan_filepath: Filepath to look for the scans in
save_path : Path to save the numpy arrs to
"""
def pairsToData(info_filepath, scan_filepath, save_path):
	merged = pd.read_excel(info_filepath)

	x_sp, x_hip, y = [], [], []
	spines, hips, bone_ages = merged['Name'] + merged['P_Spine'], merged['Name'] + merged['P_Hip'], merged['BA']

	for i, bone_age in enumerate(bone_ages):
		if pd.isnull(bone_age) or pd.isnull(spines[i]) or pd.isnull(hips[i]):
			continue

		sp_img = cv2.imread(os.path.join(scan_filepath, spines[i])  + '.jpg', cv2.IMREAD_GRAYSCALE)
		hip_img = cv2.imread(os.path.join(scan_filepath, hips[i])  + '.jpg', cv2.IMREAD_GRAYSCALE)

		if sp_img is None or hip_img is None or not len(sp_img.shape) or not len(hip_img.shape):
			continue

		sp_img = np.reshape(sp_img, (sp_img.shape[0], sp_img.shape[1], 1))
		hip_img = np.reshape(hip_img, (hip_img.shape[0], hip_img.shape[1], 1))

		x_sp.append(sp_img)
		x_hip.append(hip_img)
		y.append(bone_age)
		print(i) # marker for progress

	x_sp, x_hip, y_paired = np.array(x_sp, copy=False), np.array(x_hip, copy=False), np.array(y, copy=False)
	np.save(os.path.join(save_path, 'x_spine'), x_sp_paired)
	np.save(os.path.join(save_path, 'x_hip'), x_hip_paired)
	np.save(os.path.join(save_path, 'y_paired'), y_paired)


"""
Splits the data into training, cross-validation, and testing sets

x : List of input data, all same length
train_amt : The portion of data to be sectioned to training set
val_amt : The portion of data to be sectioned to cross-val set 
"""
def split_data(x, train_amt=0.8, val_amt=0):
	inputs = []
	tot_len = x[-1].size

	train_limit = int(tot_len * train_amt)
	val_limit = train_limit + int(tot_len * val_amt)

	for inp in x:
		x_val, x_test, x_train = inp[train_limit:val_limit], inp[val_limit:], inp[:train_limit]
		inputs.append((x_val, x_test, x_train))

	return inputs


"""
Pairs scans and cleans the data, placing them into a new csv file

file :  File names of images to be read
types : The tpyes of scans corresponding to image file
targets : Target values for the images

NOTE : Only call this to create the cleaned csv file for image extraction, NOT in fetch_data
"""
def pairScans(files, types, targets):
	spines, spines_p, hips, hips_p, spine_age, hip_age = [], [], [], [], [], []

	for i in range(len(files)):
		if types[i] == 'RightHip':
			hips.append(files[i][:13])
			hips_p.append(files[i][13:])
			hip_age.append(targets[i])
		else:
			spines.append(files[i][:13])
			spines_p.append(files[i][13:])
			spine_age.append(targets[i])

	left = pd.DataFrame({'Name': spines, 'P_Spine': spines_p, 'BA': spine_age})
	right = pd.DataFrame({'Name': hips, 'P_Hip': hips_p, 'BA': hip_age})
	merged = pd.merge(left, right, how='outer')

	has18, lastID = False, merged['Name'][0][9:]
	drop_indices = []
	for index, row in merged.iterrows():
		if index == 0:
			continue
		curID = row['Name'][9:]
		if curID != lastID:
			has18 = False	
		if pd.isnull(row['BA']) or (has18 and int(row['BA']) >= 18) or int(row['BA']) > 18:
			drop_indices.append(index)
		if not pd.isnull(row['BA']) and int(row['BA']) >= 18:
			has18 = True
		lastID = curID

	indices_to_keep = set(range(merged.shape[0])) - set(drop_indices)
	sliced = merged.take(list(indices_to_keep))
	sliced.reset_index(inplace=True)

	writer = pd.ExcelWriter('C:\\Users\\alishelton\\Documents\\PatientData\\new_cleaned_imageset.xlsx')
	sliced.to_excel(writer,'Sheet1')
	writer.save()


"""
Records the training, cross-validation, and testing data for single scans

data : A tuple/list containing the training, cross-val, and test pfile lists
output_path : Path to write the data to
"""
def recordSingle(data, output_path):
	x_data, y_data = data
	val, test, train = x_data
	val, test, train = np.array(val), np.array(test), np.array(train)
	y_val, y_test, y_train = y_data
	data_dict = {'train': train, 'train_age': y_train, \
	'cross_val': val, 'cross_val_age': y_val, 'test': test, 'test_age': y_test}
	df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
	writer = pd.ExcelWriter(output_path)
	df.to_excel(writer,'Sheet1')
	writer.save()


"""
Records the training, cross-validation, and testing data for pairs of scans

data : A tuple/list containing the training, cross-val, and test pfile lists
output_path : Path to write the data to
"""
def recordPairs(data, output_path):
	spines, hips, y_data = data
	spine_val, spine_test, spine_train = spines
	hip_val, hip_test, hip_train = hips
	val, test, train = np.array(val), np.array(test), np.array(train)
	y_val, y_test, y_train = y_data
	data_dict = {'spine_train': spine_train, 'hip_train': hip_train, 'train_age': y_train, \
	'spine_cross_val': spine_val, 'hip_cross_val': hip_val, 'cross_val_age': y_val, 'spine_test': spine_test, \
	'hip_test': hip_test,  'test_age': y_test}
	df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
	writer = pd.ExcelWriter(output_path)
	df.to_excel(writer,'Sheet1')
	writer.save()

"""
Shuffles the inital data 

path : Path of the excel file containing data to be shuffled
output_path : Path to save the new data to
"""
def shuffleDataFrame(path, output_path):
	df = pd.read_excel(path)
	df = df.sample(frac=1)
	df.reset_index(drop=True, inplace=True)
	writer = pd.ExcelWriter(output_path)
	df.to_excel(writer,'Sheet1')
	writer.save()

"""
Creates a generator for paired images

x1 : The first set of images in the input set 
x2 : The second set of images in the input set 
y : The target output values/classes
include_y : Determines whether the output should also yield y
"""
def combo_generator(x1, x2, y, include_y=True):
	datagenHip = ImageDataGenerator(featurewise_center=True, \
    	featurewise_std_normalization=True)
	datagenSpine = ImageDataGenerator(featurewise_center=True, \
    	featurewise_std_normalization=True)
	datagenSpine.fit(x1)
	datagenHip.fit(x2)
	genSpine = datagenSpine.flow(x1, y, batch_size=32, shuffle=False)
	genHip = datagenHip.flow(x2, batch_size=32, shuffle=False)
	while 1:
		x1n = genSpine.next()
		x2n = genHip.next()
		if include_y:
			yield [x1n[0], x2n], x1n[1]
		else:
			yield [x1n[0], x2n]


"""
Creates a generator that inputs demographic data in addition to images

"""
def demo_generator(x, age, sex, y, include_y=True):
	datagen = ImageDataGenerator(featurewise_center=True, \
    	featurewise_std_normalization=True)
	datagen2 = ImageDataGenerator()
	datagen3 = ImageDataGenerator()
	datagen.fit(x)
	gen = datagen.flow(x, y, batch_size=32, shuffle=False)
	gen2 = datagen2.flow(x, age, batch_size=32, shuffle=False)
	gen3 = datagen3.flow(x, sex, batch_size=32, shuffle=False)
	while 1:
		xn = gen.next()
		x2n = gen2.next()
		x3n = gen3.next()
		if include_y:
			yield [xn[0], x2n[1], x3n[1]], xn[1]
		else:
			yield [xn[0], x2n[1], x3n[1]]


"""
Formats the data output of fetch_data for input to the model, returns data in form
((x_train, y_train), (x_val, y_val), (x_test, y_test))

fetched : The output from fetch_data
"""
def format_inputs(fetched):
	x_data, y_data = fetched
	x_vals, x_tests, x_trains = [], [], []
	for inp in x_data:
		x_vals.append(inp[0])
		x_tests.append(inp[1])
		x_trains.append(inp[2])

	if len(x_vals) == 1:
		x_vals, x_tests, x_trains = x_vals[0], x_tests[0], x_trains[0]

	y_val, y_test, y_train = y_data
	return ((x_trains, y_train), (x_vals, y_val), (x_tests, y_test))

"""
Takes demographic data excel and saves to array
"""
def demographics_to_array(demo_path, output_path):
	df = pd.read_excel(demo_path)
	real_age = df['Real_Age']
	sex = df['Sex']
	weight = df['Weight']
	height = df['Height']
	np.save(os.path.join(output_path, 'real_age'), real_age)
	np.save(os.path.join(output_path, 'sex'), sex)
	np.save(os.path.join(output_path, 'weight'), weight)
	np.save(os.path.join(output_path, 'height'), height)



############# DEPRECATED ################
"""
Returns a dictionary of the information in a dataframe holding the file, target, and
extraneous information

df : The dataframe holding pfile names, age, sex, etc.
"""
def extractSingle(df):
	files = df['pfiles'].values
	ids = df['IDNO'].values
	files = [pfile[0:8] + '_' + str(id_num) + pfile[8:] for pfile, id_num in zip(files, ids)]
	types = df['types'].values
	sexes = df['sexes'].values
	bone_age = df['YEAR'].values + (df['MONTH'].values / 12)
	files, types, sexes, bone_age = shuffleData((files, types, sexes, bone_age))
	return dict(zip(files, zip(types, sexes, bone_age)))


"""
Returns the data as it was passed in, shuffled randomly

data : A tuple/list of corresponding entries to shuffle
"""
def shuffleData(data):
	combined = list(zip(*data))
	random.shuffle(combined)
	return zip(*combined)

"""
Normalizes input numpy images

img : Numpy array of images to be normalized
"""
def normalize(img):
	mean = np.mean(img, axis=(0, 1), dtype=np.float32)
	std = np.std(img, axis=(0, 1), dtype=np.float32)
	return (img - mean) / std

#singleScanToData('Hip', 'C:\\Users\\alishelton\\Documents\\PatientData\\shuffled_cleaned_imageset.xlsx', \
#	'C:\\Users\\alishelton\\Documents\\DistributedScans', 'C:\\Users\\alishelton\\Documents\\numpy_arrs\\limited', 450)

