import random
import numpy as np
import pandas as pd
import pickle
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
    




# Hyperparameter
batch_size = 128


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





# Define the data loader for training
train_dataset = CustomDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the data loader for testing
test_dataset = CustomDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, x1, x2):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0,c0))
        
        out = self.fc(out[:, -1, :])
        
        h01 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(x1.device)
        c01 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(x1.device)
        
        out1, _ = self.lstm(x1, (h01,c01))
        
        out1 = self.fc(out1[:, -1, :])

        h02 = torch.zeros(self.num_layers, x2.size(0), self.hidden_size).to(x2.device)
        c02 = torch.zeros(self.num_layers, x2.size(0), self.hidden_size).to(x2.device)
        
        out2, _ = self.lstm(x2, (h02,c02))
        
        out2 = self.fc(out2[:, -1, :])
        return out, out1, out2

# Initialize the model
net = LSTMModel(input_size=channel, hidden_size=64, num_layers=1, output_size=128)
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
        anchor, positive, negative = net(x, x1, x2)
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
            anchor, positive, negative = net(x, x1, x2)

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
        anchor, positive, negative = net(x, x1, x2)
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
        anchor, positive, negative = net(x, x1, x2)
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
    print(f'Lstm WISDM Triplet accuracy: {accuracy}')  
else:
    print(f'Lstm META_HAR Triplet accuracy: {accuracy}')  


