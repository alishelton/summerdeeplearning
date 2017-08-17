"""
This file is used for quick analysis and plotting of model performance


Author : Ali Shelton
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.formula.api as sm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse
from scipy import stats
import pandas as pd


test_act_path = 'C:\\Users\\alishelton\\Documents\\analysis_limited_hip\\y_test.txt'
test_pred_path = 'C:\\Users\\alishelton\\Documents\\analysis_limited_hip\\y_pred.txt'
val_act_path = 'C:\\Users\\alishelton\\Documents\\analysis_distributed_paired\\y_val.txt'
val_path = 'C:\\Users\\alishelton\\Documents\\analysis_distributed_paired\\y_val_pred.txt'


"""
Plots the predicted bone ages against the expected bone ages

predict_path: Path to the predicted ages array
actual_path: Path to the actual ages array
pred_type: The set type, either Cross Validation or Testing
"""
def plotPreds(predict_path, actual_path, pred_type):
	y = np.loadtxt(predict_path)
	x = np.loadtxt(actual_path)
	xn = x.reshape(x.shape[0], 1)
	yn = y.reshape(y.shape[0], 1)

	data={'x': x, 'y': y}
	results = sm.ols(formula="y ~ x", data=data).fit()
	print(results.summary())
	print('p-vals: ')
	print(results.pvalues)

	xy = np.vstack([x, y])
	z = stats.gaussian_kde(xy)(xy)
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	fig, ax = plt.subplots()
	cax = ax.scatter(x, y, c=z, s=50, edgecolor='')
	ax.set_ylim([3, 20])
	ax.set_xlim([3, 20])
	#fig.colorbar(cax)
	intercept, coef = results.params
	plt.plot(x, coef*x + intercept, '-', color='black')
	plt.plot(x, x, '-', color='red')
	
	plt.ylabel('Predicted Bone Age')
	plt.xlabel('Actual Bone Age')
	plt.title(pred_type + ' Prediction vs. Actual Bone Age')
	plt.show()

"""
Plots a histogram of the dataset

data_path : Path to excel spreadsheet with bone age info
dataset: Type of dataset, among Training, Cross Validation, and Testing
"""
def plotDataHist(data_path, dataset):
	df = pd.read_excel('C:\\Users\\alishelton\\Documents\\PatientData\\new_cleaned_imageset.xlsx')
	y_data = df['BA'].values
	y = [d for d in y_data if not pd.isnull(d)]
	
	tot_len = len(y)
	train_limit = int(tot_len * 0.64)
	val_limit = train_limit + int(tot_len * 0.16)

	y_val, y_test, y_train = y[train_limit:val_limit], y[val_limit:], y[:train_limit]

	plt.title('Count of Bone Ages in the ' + dataset + ' Dataset')
	plt.xlabel('Bone Age')
	plt.ylabel('Number of Cases')
	if dataset == 'Training':
		h_type = y_train
	elif dataset == 'Cross Validation':
		h_type = val
	else:
		h_type = test

	# h = 2 * stats.iqr(h_type) * math.pow(h_type.size, -(1/3))
	num_bins = 14 # int((np.max(h_type) - np.min(h_type))/ h)

	hist, bins = np.histogram(h_type, bins=num_bins)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center', width=width)
	plt.show()

"""
Plots a histogram of the error on the dataset

predict_path: Path to the predicted ages array
actual_path: Path to the actual ages array
"""
def plotErrorHist(predict_path, actual_path):
	preds = np.loadtxt(predict_path)
	actual = np.loadtxt(actual_path)
	diff = actual - preds

	h = 2 * stats.iqr(diff) * math.pow(diff.size, -(1/3))
	num_bins = int((np.max(diff) - np.min(diff))/ h)

	plt.title('ResNet 34 Performance')
	plt.xlabel('Difference from Bone Age in Years')
	plt.ylabel('Number of Cases')
	hist, bins = np.histogram(diff, bins=num_bins)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center', width=width)
	plt.show()

"""
Plots the error values over time

error_type: The type of error, a string, either Cross Validation or Training
yvals: The error values over time
"""
def plotError(error_type, yvals):
	plt.plot(yvals)
	plt.title('Mean Absolute Error of the' + error_type + ' Set Over Each Epoch')
	plt.ylabel('Mean Absolute Error')
	plt.xlabel('Epoch Number')
	plt.show()

"""
Plots the loss values over time

loss_type: The type of loss, a string, either Cross Validation or Training
yvals: The loss values over time
has_log: Boolean, whether or not y should be log scaled 
is_batch : Whether or not the data is in batches
"""
def plotLoss(loss_type, yvals, has_log, is_batch):
	plt.plot(yvals)
	plt.title(loss_type + ' Set Loss Over Each Epoch')
	y_label = 'Loss (Mean Squared Error)'
	if has_log:
		y_label = 'Log ' + y_label
		plt.yscale('log')
	plt.ylabel(y_label)
	x_label = 'Epoch Number'
	if is_batch:
		x_label = 'Batch Number'
	plt.xlabel(x_label)
	plt.show()


plotPreds(test_pred_path, test_act_path, 'Test')
# plotDataHist('C:\\Users\\alishelton\\Documents\\PatientData\\shuffled_cleaned_imageset.xlsx', 'Training')