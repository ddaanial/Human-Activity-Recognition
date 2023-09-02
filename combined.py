import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
import pickle
import os
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('channel', type=int)
parser.add_argument('num_epoch_normal', type=int)
parser.add_argument('num_epoch_pretraining', type=int)
parser.add_argument('lr_normal', type=float)
parser.add_argument('lr_pretraining', type=float)
parser.add_argument('temperature', type=float)
parser.add_argument('device', type=str)
args = parser.parse_args()

channel = args.channel
num_epoch_normal = args.num_epoch_normal
num_epoch_pretraining = args.num_epoch_pretraining
lr_normal = args.lr_normal
lr_pretraining = args.lr_pretraining
temperature = args.temperature
device = args.device




def get_cwt(index):

    # set up wavelet parameters
    wavelet = 'morl'
    scales = np.arange(1, 51)

    # perform CWT and get power spectrum for each component
    power = []
    for i in range(channel):
        coef, freqs = pywt.cwt(X[index][i], scales, wavelet)
        power.append(abs(coef)**2)

    return np.array(power)

if channel == 3:
    with open('WISDM.pkl', 'rb') as f:
        X, y = pickle.load(f)
else:
    with open('Meta_Har.pkl', 'rb') as f:
        X, y = pickle.load(f)

# Convert data and labels to NumPy arrays
X = np.array(X)
y = np.array(y)


X_ = []
for idx in range(len(X)):
    X_.append(get_cwt(idx))

X_cwt = np.array(X_)
del X_


# define the ResNet-18 model with skip connections
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.shortcut(residual)
        x = F.relu(x)

        return x

class ResNet18(torch.nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18, self).__init__()
        self.conv1 = torch.nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = torch.nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.layer2 = torch.nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer3 = torch.nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        self.layer4 = torch.nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

# Define lstm architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x, (h0,c0))
        
        out = self.fc(out[:, -1, :])
        
        return out

# Define lstm architecture
class model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = LSTMModel(input_size=channel, hidden_size=64, num_layers=1, output_size=512)
        self.resnet = ResNet18()
        self.lin_layer = nn.Sequential(
            nn.Linear(2*channel, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        self.linear1 = nn.Linear(3*512, 256)
        self.linear2 = nn.Linear(256, 6)
        
    def forward(self, x_lstm, x_cwt):
        x_lstm_out = self.lstm(x_lstm)
        # Calculate the mean along the second dimension (time steps)
        mean_values = x_lstm.permute(0, 2, 1).mean(dim=1)  # Shape: (5998, 3)

        # Calculate the variance along the second dimension (time steps)
        variance_values = x_lstm.permute(0, 2, 1).var(dim=1)  # Shape: (5998, 3)

        # Concatenate mean and variance tensors along the last dimension
        x_stat = torch.cat((mean_values, variance_values), dim=1)  # Shape: (5998, 6)
        x_stat = self.lin_layer(x_stat)

        x_cwt = self.resnet(x_cwt)

        combined_tensor = torch.cat((x_lstm_out, x_cwt, x_stat), dim=1)

        out = self.linear1(combined_tensor)
        out = self.linear2(out)

        return out
# create DataLoader for training set

X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train_cwt, X_test_cwt, y_train, y_test = train_test_split(X_cwt, y, test_size=0.2, random_state=123)
del X
del X_cwt
del y
# Split data into train and test sets


# Convert data and labels to PyTorch tensors
X_train_lstm = torch.from_numpy(X_train_lstm).float()
X_train_cwt = torch.from_numpy(X_train_cwt).float()
y_train = torch.from_numpy(y_train).long()
X_test_lstm = torch.from_numpy(X_test_lstm).float()
X_test_cwt = torch.from_numpy(X_test_cwt).float()
y_test = torch.from_numpy(y_test).long()




# Hyperparameters
batch_size = 32
#learning_rate = 0.001
#num_epochs = num_epoch_
input_dim = channel

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initialize model, loss function, and optimizer
model = model(input_size=channel, hidden_size=64, num_layers=1, output_size=128)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_normal)



best_train_loss = float('inf')
for epoch in range(num_epoch_normal):
    epoch_loss = 0
    epoch_acc = 0
    
    for i in range(0, len(X_train_lstm), batch_size):
        inputs_lstm = X_train_lstm[i:i+batch_size].to(device)
        inputs_cwt = X_train_cwt[i:i+batch_size].to(device)
        targets = y_train[i:i+batch_size].to(device)

        outputs = model(inputs_lstm, inputs_cwt)
        loss = criterion(outputs, targets)

        # Save the best model so far
        if loss < best_train_loss:
            best_train_loss = loss
            best_model_state = model.state_dict()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        epoch_acc += accuracy_score(predicted.cpu().numpy(), targets.cpu().numpy()) * inputs_lstm.shape[0]

    epoch_loss /= len(X_train_lstm) / batch_size
    epoch_acc /= len(y_train)


    # print(f'Epoch {epoch+1}, loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}')
    # # Open a file for writing
    # if channel == 3:
    #     with open('Results/combined/WISDM/WISDM_Loss.txt', 'a') as file:
    #         # Write the accuracy 
    #         file.write(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}\n')
    # else:
    #     with open('Results/combined/Meta_Har/Meta_Har_Loss.txt', 'a') as file:
    #         # Write the accuracy 
    #         file.write(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}\n')

# Save the best model state to a file
# if channel == 3:
#     torch.save(best_model_state, 'Results/combined/WISDM/WISDM_best_model.pth')
# else:
#     torch.save(best_model_state, 'Results/combined/Meta_Har/Meta_Har_best_model.pth')


model.load_state_dict(best_model_state)


preds = []
# Evaluate model on test set
with torch.no_grad():
    model.eval()
    test_loss = 0
    test_acc = 0
    
    for i in range(0, len(X_test_lstm), batch_size):
        inputs_lstm = X_test_lstm[i:i+batch_size].to(device)
        inputs_cwt = X_test_cwt[i:i+batch_size].to(device)
        targets = y_test[i:i+batch_size].to(device)

        outputs = model(inputs_lstm, inputs_cwt)
        loss = criterion(outputs, targets)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_acc += accuracy_score(predicted.cpu().numpy(), targets.cpu().numpy()) * inputs_lstm.shape[0]
        preds.extend(list(predicted.cpu().numpy()))

    test_loss /= len(X_test_lstm) / batch_size
    test_acc /= len(X_test_lstm)

    f1 = f1_score(preds,y_test,average='weighted')

#     if channel == 3:
#         with open('Results/combined/WISDM/WISDM_Result.txt', 'w') as file:
#             # Write the accuracy and loss to the file
#             file.write(f'test loss: {test_loss}\n')
#             file.write(f'test accuracy: {test_acc}\n')
#             file.write(f'F1 Score: {f1}\n')
#     else:
#         with open('Results/combined/Meta_Har/Meta_Har_Result.txt', 'w') as file:
#             # Write the accuracy and loss to the file
#             file.write(f'test loss: {test_loss}\n')
#             file.write(f'test accuracy: {test_acc}\n')
#             file.write(f'F1 Score: {f1}\n')

if channel == 3:
    print(f'Combined WISDM CE accuracy: {test_acc}')  
else:
    print(f'Combined META_HAR CE accuracy: {test_acc}')   
# print(f'Test loss: {test_loss:.4f}, test accuracy: {test_acc:.2%}, F1 Score: {f1}')

# if channel == 3:
#     labels = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
# else:
#     labels = ['Walk', 'Bike', 'Upstairs', 'Downstairs', 'Run', 'bus/taxi']

# # Compute confusion matrix
# cm = confusion_matrix(y_test, preds)
# # normalize confusion matrix
# cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# # Plot confusion matrix
# fig, ax = plt.subplots()
# im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# ax.figure.colorbar(im, ax=ax)
# ax.set(xticks=np.arange(cm.shape[1]),
#     yticks=np.arange(cm.shape[0]),
#     xticklabels=labels, yticklabels=labels,
#     ylabel='Actual label',
#     xlabel='Predicted label')

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(cm_norm.shape[0]):
#     for j in range(cm_norm.shape[1]):
#         ax.text(j, i, format(cm_norm[i, j], '.2f'),
#                 ha="center", va="center",
#                 color="black" if cm_norm[i, j] > cm_norm.max() / 2. else "black")
        
# fig.tight_layout()

# if channel == 3:
#     plt.savefig('Results/combined/WISDM/WISDM_confusion_matrix.png')
# else:
#     plt.savefig('Results/combined/Meta_Har/Meta_Har_confusion_matrix.png')

