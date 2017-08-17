import os
from shutil import copyfile
import pandas as pd

####### Put information here #######

# The file path of the text file with the file names
filepath = ''

# The original directory to move files out of 
original_path = ''

# The destination directory to move files to
destination_path = ''

######################################################################################################################################################

"""
Grabs a list of files from excel sheet and returns them in an array

filepath : file path to the excel file with pfile names
"""
def grabFileList(filepath):
	df = pd.read_excel(filepath)
	return df['pfile_name2']


"""
Finds the file in the path from a higher directory

name : file name to search for
path : higher directory to search for the file from
"""
def findAll(path):
	cache = {}
	for root, dirs, files in os.walk(path):
		for file in files:
			cache[file.lower()] = os.path.join(root, file)
	return cache


"""
Moves the files from text_file from the original path to their destination

text_file : text file containing name of files to move
orig_path : Original path to move files from
desintation_path : Destination path to move files to
"""
def moveFiles(ex_path, orig_path, dest_path):
	# Grab the file names from text_file
	file_path_cache = findAll(orig_path)
	for file in grabFileList(ex_path):
		# Find the file path of the file
		try:
			file_path = file_path_cache[file.lower()]
		except:
			pass

		#Don't move if not found
		if file_path == None:
			print('File: ' + file + ' could not be moved')
			continue

		# Copy file into desination directory
		copyfile(file_path, os.path.join(dest_path, file.lower()))


# Call the function
moveFiles(filepath, original_path, destination_path)

