import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
import matplotlib.pyplot as plt

data_dir = "Data/Meta-HAR-data"

# Seprating accelerometer and gyroscope signals.
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


# Reading file and create the data structure.
X, y = [], []
files = [file for file in os.listdir(data_dir) if "txt" in file]
for filename in files:
    data = open(os.path.join(data_dir, filename))
    for line in data:
        _, act, acc, gyro = line.strip().split("\t")
        signal = np.concatenate((data_process_sensor(acc), data_process_sensor(gyro)), axis=0)
        signal = signal.T
        X.append(signal)
        y.append(act)
X = np.array(X)
y = np.array(y)

# X shape = (39168, 150, 6) 
# y shape = (39168,)


### Visualization

X_2d = np.array([features_2d.flatten() for features_2d in X])
y_ = y.astype(int).tolist()

# Reduction with LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_2d, y_)

# Plot the result
x1 = X_lda[:,0]
x2 = X_lda[:,1]
plt.scatter(x1, x2, c=y_, edgecolor='none', alpha=.8, cmap=plt.colormaps['viridis'])
plt.xlabel('lda1')
plt.ylabel('lda2')
plt.colorbar()
plt.axis('off')
plt.savefig('LDA_Recuction.png')
