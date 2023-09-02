import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Data Dicrectory
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


# Replacing original label with new label
d = {'1':0, '2':1, '3':2, '4':3, '5':4, '7':5}
# 0 --> Walk
# 1 --> Bike
# 2 --> Upstairs
# 3 --> Downstairs
# 4 --> Run
# 5 --> bus/taxi



# Reading file and create the data structure.
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

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Define the hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32


# Define the convolutional neural network model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 36, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 36)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model
model = ConvNet()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        # Get a batch of data
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]
        labels = labels.long()
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs,labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the training loss after each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Evaluate the model on your testing data
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / total
    f1 = f1_score(predicted,y_test,average='weighted')
    print("F1 Score:", f1)
    print('Test Accuracy:', accuracy)


# labels
labels = ['Walk', 'Bike', 'Upstairs', 'Downstairs', 'Run', 'bus/taxi']

# Compute confusion matrix
cm = confusion_matrix(y_test, predicted)

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels,
       ylabel='Actual label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2. else "black")
        
fig.tight_layout()
plt.show()