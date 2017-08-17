import pandas as pd
import numpy as np
"""
shuffled = pd.read_excel('C:\\Users\\alishelton\\Documents\\PatientData\\shuffled_cleaned_imageset.xlsx')

index, ba, name, sp, hip = shuffled['index'], shuffled['BA'], shuffled['Name'], shuffled['P_Spine'], shuffled['P_Hip']
file_match = name + hip
file_match = file_match.values
# new_frame = pd.DataFrame({'index': index, 'BA': ba, 'Name': name, 'P_Spine': sp, 'P_Hip': hip, 'pfiles': new_col})
all_data = pd.read_excel('C:\\Users\\alishelton\\Documents\\PatientData\\distributed_patient_data.xlsx')
pfiles, weight, height, age, sex, ids = all_data['pfiles'].values, all_data['WEIGHT'].values, all_data['HEIGHT'].values, \
	all_data['AGE'].values, all_data['sexes'].values, all_data['IDNO'].values

pfiles = [pfile[0:8] + '_' + str(id_num) + pfile[8:] for pfile, id_num in zip(pfiles, ids)]

info_dict = dict(zip(pfiles, zip(weight, height, age, sex)))
new_weight, new_height, new_age, new_sex = [], [], [], []
new_index, new_ba, new_name, new_sp, new_hip = [], [], [], [], []
i = 0
for file in file_match:
	if pd.isnull(file):
		continue
	else:
		l = info_dict[file]
	new_index.append(index[i])
	new_ba.append(ba[i])
	new_name.append(name[i])
	new_sp.append(sp[i])
	new_hip.append(hip[i])
	new_weight.append(l[0])
	new_height.append(l[1])
	new_age.append(l[2])
	new_sex.append(l[3])
	print(i)
	i += 1


final_frame = pd.DataFrame({'index': new_index, 'BA': new_ba, 'Name': new_name, 'P_Spine': new_sp, 'P_Hip': new_hip, \
	'Height': new_height, 'Weight': new_weight, 'Real_Age': new_age, 'Sex': new_sex})

writer = pd.ExcelWriter('C:\\Users\\alishelton\\Documents\\PatientData\\shuffled_cleaned_demographic2.xlsx')
final_frame.to_excel(writer,'Sheet1')
writer.save()
"""
sex = np.load('C:\\Users\\alishelton\\Documents\\numpy_arrs\\demographics\\sex.npy')
np.place(sex, sex=='M', 0)
np.place(sex, sex=='F', 1)
np.save('C:\\Users\\alishelton\\Documents\\numpy_arrs\\demographics\\sex2', sex)
