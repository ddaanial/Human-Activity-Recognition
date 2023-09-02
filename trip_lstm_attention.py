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


# Hyperparameters
batch_size = 32
#learning_rate = 0.001
#num_epochs = num_epoch
input_dim = channel
hidden_dim = 64
inner_dim = 128
output_dim = 128


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



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)

        return x

class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Attention layers
        self.attention = Attention(dim = hidden_dim)

        # FC
        self.fc_1 = nn.Linear(hidden_dim, inner_dim)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(inner_dim, output_dim)
        

        
    def forward(self, x, x1, x2):
        # Pass input through LSTM layers
        output, (h_n, c_n) = self.lstm(x)

        x= self.attention(output)
        x_avg = torch.mean(x, dim=1)
        
        # Apply linear transformation to the averaged output
        x = self.relu(self.fc_1(x_avg))

        #____________
        # Pass input through LSTM layers
        output1, (h_n1, c_n1) = self.lstm(x1)

        x1= self.attention(output1)
        x_avg1 = torch.mean(x1, dim=1)
        
        # Apply linear transformation to the averaged output
        x1 = self.relu(self.fc_1(x_avg1))
         #____________
        # Pass input through LSTM layers
        output2, (h_n2, c_n2) = self.lstm(x2)

        x2= self.attention(output2)
        x_avg2 = torch.mean(x2, dim=1)
        
        # Apply linear transformation to the averaged output
        x2 = self.relu(self.fc_1(x_avg2))


        return x, x1, x2

# Initialize the model
net = AttentionModel(input_dim=channel, hidden_dim=hidden_dim, num_layers=2, output_dim=128)
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
    print(f'Lstm Attention WISDM Triplet accuracy: {accuracy}')  
else:
    print(f'Lstm Attention META_HAR Triplet accuracy: {accuracy}')  

