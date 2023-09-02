import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from math import log
from torch.autograd import Function
import torch.nn.functional as F
from pytorch_metric_learning import losses

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

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=temperature):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=temperature)(logits, torch.squeeze(labels))

if channel == 3:
    with open('WISDM.pkl', 'rb') as f:
        X, y = pickle.load(f)
else:
    with open('Meta_Har.pkl', 'rb') as f:
        X, y = pickle.load(f)

# Convert data and labels to NumPy arrays
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Split data into train and test sets
if channel == 3:
    X_train = X_train.reshape((-1, 150, 3))
    X_test = X_test.reshape((-1, 150, 3))
else:
    X_train = X_train.reshape((-1, 150, 6))
    X_test = X_test.reshape((-1, 150, 6))

# Convert data and labels to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()




# Hyperparameters
batch_size = 32
input_dim = channel
hidden_dim = 150
output_dim = 128


# Define the convolutional neural network model

class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Attention layers
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Pass input through LSTM layers
        output, (h_n, c_n) = self.lstm(x)
        
        # Compute attention weights
        alpha = F.softmax(self.attention_layer(output), dim=1)
        
        # Compute attention output
        attention_output = torch.sum(alpha * output, axis=1)

        # Pass attention output through output layer
        output = self.output_layer(attention_output)
        
        return output

# Initialize the model
model = AttentionModel(input_dim=channel, hidden_dim=hidden_dim, num_layers=2, output_dim=128)
model.to(device)
# Define the loss function and optimizer
criterion = SupervisedContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_pretraining)

best_train_loss = float('inf')
# Train the model
for epoch in range(num_epoch_pretraining):
    for i in range(0, len(X_train), batch_size):
        # Get a batch of data
        inputs = X_train[i:i+batch_size].to(device)
        labels = y_train[i:i+batch_size].to(device)

        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs,labels)

        # Save the best model so far
        if loss < best_train_loss:
            best_train_loss = loss
            best_model_state = model.state_dict()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

 

model.load_state_dict(best_model_state)


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
classifier = ClassificationModel()
classifier.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr_normal)

best_train_loss = float('inf')
# Train the model
for epoch in range(num_epoch_normal):
    # Set the model to training mode
    classifier.train()

    # Iterate over the batches
    for i in range(0, len(X_train), batch_size):
        # Zero the gradients
        optimizer.zero_grad()

        # Get a batch of data
        inputs = X_train[i:i+batch_size].to(device)
        labels = y_train[i:i+batch_size].to(device)

        # Forward pass
        logits = model(inputs)
        outputs = classifier(logits)
        labels = labels.long()
        loss = criterion(outputs, labels)

        # Save the best model so far
        if loss < best_train_loss:
            best_train_loss = loss
            best_model_state =classifier.state_dict()

        # Backward pass
        loss.backward()
        optimizer.step()



classifier.load_state_dict(best_model_state)
# Evaluate the model on the test set
classifier.eval()
preds = []
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        # Get a batch of data
        inputs = X_test[i:i+batch_size].to(device)
        labels = y_test[i:i+batch_size].to(device)
        logits = model(inputs)
        outputs = classifier(logits)
        labels = labels.long()

        _, predicted = torch.max(outputs.data, 1)
        total += predicted.size(0)
        correct += (predicted == labels).sum().item()

        preds.extend(list(predicted.cpu().numpy()))
    accuracy = correct / total
if channel == 3:
    print(f'Lstm Linear WISDM Contrastive accuracy: {accuracy}')  
else:
    print(f'Lstm Linear META_HAR Contrastive accuracy: {accuracy}')    



