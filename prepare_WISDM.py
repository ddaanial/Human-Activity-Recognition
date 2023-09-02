import csv
import numpy as np
import pandas as pd
import os
import torch
import pickle

def makeit(sub_data):
    padding_value = [0, 0, 0]
    padded_list = []

    # Determine the number of groups required
    num_groups = len(sub_data) // 150 + (len(sub_data) % 150 != 0)

    # Pad each sublist in my_list and add it to the padded_list
    for i in range(num_groups * 150):
        if i < len(sub_data):
            current_sublist = sub_data[i]
        else:
            current_sublist = padding_value
        padded_list.append(current_sublist)

    # Reshape the padded_list into groups of size 150 x 3
    final_list = [padded_list[i:i+150] for i in range(0, len(padded_list), 150)]
    return final_list

current_activity = None
with open('Data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = []
    temp = []
    labels = []
    for row in reader:
        if row ==[]:
            continue
        row = [x[:-1] if idx==5 else x for idx, x in enumerate(row)]
        if row[1] !=current_activity:
            data.append(temp)
            labels.append(current_activity)
            temp = []
        current_activity = row[1]
        temp.append([row[i] for i in [3,4,5]])
Data = data[1:]
labels = labels[1:]

new_label = list()
flag = 0
for label, data in zip(labels, Data):
    temp = np.array(makeit(data))
    if flag == 0:
        new_list = temp
        flag = 1
    else:
        new_list = np.concatenate((new_list, temp), axis=0)
    for i in range(np.array(temp).shape[0]):
        new_label.append(label)
new_label = np.array(new_label)
new_list = np.transpose(new_list, axes=(0, 2, 1))

dic_label = {
    'Downstairs': 0,
    'Jogging': 1,
    'Sitting': 2,
    'Standing': 3,
    'Upstairs': 4,
    'Walking': 5
}
# 0 --> Downstairs
# 1 --> Jogging
# 2 --> Sitting
# 3 --> Standing
# 4 --> Upstairs
# 5 --> Walking

new_label = np.array([dic_label[el] for el in new_label])
new_list = np.where(new_list == '', 0, new_list)

X = torch.tensor(new_list.astype(float)).float()
y = torch.tensor(new_label).float()


# save the tensors using pickle
with open('WISDM.pkl', 'wb') as f:
    pickle.dump((X, y), f)

