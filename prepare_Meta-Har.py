import numpy as np
import pandas as pd
import os
import torch
import pickle
def data_process_sensor(data_str, length=150):
    data_list = data_str.strip().split(",")
    acc_x = []
    acc_y = []
    acc_z = []
    for i in range(int(len(data_list) / 3)):
        acc_x.append(float(data_list[i * 3]))
        acc_y.append(float(data_list[i * 3 + 1]))
        acc_z.append(float(data_list[i * 3 + 2]))
    acc_x = np.array(acc_x[0:length])
    acc_y = np.array(acc_y[0:length])
    acc_z = np.array(acc_z[0:length])
    return np.stack([acc_x, acc_y, acc_z], axis=0) 

d = {'1':0, '2':1, '3':2, '4':3, '5':4, '7':5}
# 0 --> Walk
# 1 --> Bike
# 2 --> Upstairs
# 3 --> Downstairs
# 4 --> Run
# 5 --> bus/taxi

data_dir = "Data/Meta-HAR-data"

X, y = [], []
files = [file for file in os.listdir(data_dir) if "txt" in file]
for filename in files:
    data = open(os.path.join(data_dir, filename))
    for line in data:
        _, act, acc, gyro = line.strip().split("\t")
        signal = np.concatenate((data_process_sensor(acc), data_process_sensor(gyro)), axis=0)
        X.append(signal)
        y.append(act)

X = np.array(X)
y = np.array(y)

y = np.array([d[i] for i in y if i in d])
X = torch.tensor(X).float()
y = torch.tensor(y).float()

# save the tensors using pickle
with open('Meta_Har.pkl', 'wb') as f:
    pickle.dump((X, y), f)

