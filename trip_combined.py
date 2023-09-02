import random
import numpy as np
import pandas as pd
import pickle
import pywt
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

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


def make_cwt(xx):
    X_ = []
    for idx in range(len(xx)):
        X_.append(get_cwt(idx))

    X_np = np.array(X_)

    X_cwt = torch.tensor(X_np)
    return X_cwt


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, label_file):
        self.x = data_file
        self.y = label_file

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # Get the anchor sample
        x_anchor = self.x[index]
        y_anchor = self.y[index]
        # Randomly choose a positive sample from the same class as the anchor sample
        indices = np.where(self.y == self.y[index])[0]
        index_positive = np.random.choice(indices)
        x_positive = self.x[index_positive]

        # Randomly choose a negative sample from a different class than the anchor sample
        indices = np.where(self.y != self.y[index])[0]
        index_negative = np.random.choice(indices)
        x_negative = self.x[index_negative]

        # Stack the samples into pairs and create the label vector
        x = x_anchor
        x1 = x_positive
        x2 = x_negative
        labels = np.array([1, 0], dtype=np.float32)

        # Convert the data to PyTorch tensors
        x = torch.from_numpy(x).float()
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()

        return (x, x1, x2, y_anchor)

# Define the contrastive loss function
class TripleContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripleContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_pos = (anchor - positive).pow(2).sum(-1)
        distance_neg = (anchor - negative).pow(2).sum(-1)
        loss = torch.clamp(distance_pos - distance_neg + self.margin, min=0.0)
        return loss.mean()




# Hyperparameter
batch_size = 32

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Define the data loader for training
train_dataset = CustomDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the data loader for testing
test_dataset = CustomDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
        self.linear2 = nn.Linear(256, 128)
        
    def forward(self, x_lstm, x_cwt, x1_lstm, x1_cwt, x2_lstm, x2_cwt):
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



        x1_lstm_out = self.lstm(x1_lstm)
        # Calculate the mean along the second dimension (time steps)
        mean_values1 = x1_lstm.permute(0, 2, 1).mean(dim=1)  # Shape: (5998, 3)

        # Calculate the variance along the second dimension (time steps)
        variance_values1 = x1_lstm.permute(0, 2, 1).var(dim=1)  # Shape: (5998, 3)

        # Concatenate mean and variance tensors along the last dimension
        x_stat1 = torch.cat((mean_values1, variance_values1), dim=1)  # Shape: (5998, 6)
        x_stat1 = self.lin_layer(x_stat1)

        x1_cwt = self.resnet(x1_cwt)

        combined_tensor1 = torch.cat((x1_lstm_out, x1_cwt, x_stat1), dim=1)

        out1 = self.linear1(combined_tensor1)
        one = self.linear2(out1)


        x2_lstm_out = self.lstm(x2_lstm)
        # Calculate the mean along the second dimension (time steps)
        mean_values2 = x2_lstm.permute(0, 2, 1).mean(dim=1)  # Shape: (5998, 3)

        # Calculate the variance along the second dimension (time steps)
        variance_values2 = x2_lstm.permute(0, 2, 1).var(dim=1)  # Shape: (5998, 3)

        # Concatenate mean and variance tensors along the last dimension
        x_stat2 = torch.cat((mean_values2, variance_values2), dim=1)  # Shape: (5998, 6)
        x_stat2 = self.lin_layer(x_stat2)

        x2_cwt = self.resnet(x2_cwt)

        combined_tensor2 = torch.cat((x2_lstm_out, x2_cwt, x_stat2), dim=1)

        out2 = self.linear1(combined_tensor2)
        two = self.linear2(out2)

        return out, one, two

# Initialize the model
net = model(input_size=channel, hidden_size=64, num_layers=1, output_size=128)
net.to(device)
criterion = TripleContrastiveLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr_pretraining)

best_train_loss = float('inf')
# Train the model
for epoch in range(num_epoch_pretraining):
    # Set the model to training mode
    net.train()

    # Iterate over the batches
    for i, (x, x1, x2, labels) in enumerate(train_loader):
        x = x.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        anchor, positive, negative = net(x, make_cwt(x), x1, make_cwt(x1), x2, make_cwt(x2))
        loss = criterion(anchor, positive, negative)

        # Save the best model so far
        if loss < best_train_loss:
            best_train_loss = loss
            best_model_state =net.state_dict()

        # Backward pass
        loss.backward()
        optimizer.step()

    # Evaluate the model on the test set
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for  (x, x1, x2, labels) in test_loader:
            x = x.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            anchor, positive, negative = net(x, make_cwt(x), x1, make_cwt(x1), x2, make_cwt(x2))

            # Compute distances between anchor and positive/negative examples
            dist_pos = F.pairwise_distance(anchor, positive)
            dist_neg = F.pairwise_distance(anchor, negative)

            # Count the number of correct predictions (dist_pos < dist_neg)
            correct += torch.sum(dist_pos < dist_neg).item()
            total += dist_pos.size(0)

    accuracy = 100 * correct / total


net.load_state_dict(best_model_state)


# Create a new model for classification using the embeddings from the contrastive model
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model
model = ClassificationModel()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_normal)

best_train_loss = float('inf')
# Train the model
for epoch in range(num_epoch_normal):
    # Set the model to training mode
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    # Iterate over the batches
    for i, (x, x1, x2, labels) in enumerate(train_loader):
        x = x.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        anchor, positive, negative = net(x, make_cwt(x), x1, make_cwt(x1), x2, make_cwt(x2))
        outputs = model(anchor)
        labels = labels.long()
        loss = criterion(outputs, labels)

        # Save the best model so far
        if loss < best_train_loss:
            best_train_loss = loss
            best_model_state =model.state_dict()

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        epoch_acc += accuracy_score(predicted.cpu().numpy(), labels.cpu().numpy()) * predicted.shape[0]



    epoch_loss /= len(X_train) / batch_size
    epoch_acc /= len(y_train)



model.load_state_dict(best_model_state)
# Evaluate the model on the test set
model.eval()
preds = []
with torch.no_grad():
    correct = 0
    total = 0
    test_loss = 0
    for  (x, x1, x2, labels) in test_loader:
        x = x.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        anchor, positive, negative = net(x, make_cwt(x), x1, make_cwt(x1), x2, make_cwt(x2))
        outputs = model(anchor)
        labels = labels.long()

        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += predicted.size(0)
        correct += (predicted == labels).sum().item()
        preds.extend(list(predicted.cpu().numpy()))
    accuracy = correct / total
    test_loss /= len(X_test) / batch_size
    f1 = f1_score(preds,y_test,average='weighted')    


if channel == 3:
    print(f'Combined WISDM Triplet accuracy: {accuracy}')  
else:
    print(f'Combined META_HAR Triplet accuracy: {accuracy}')     


