import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_layer2 = nn.Linear(output_dim, 6)
        
    def forward(self, x):
        # Pass input through LSTM layers
        output, (h_n, c_n) = self.lstm(x)
        

        # linear layer
        output = self.output_layer(output)
        output = torch.mean(output, dim=1)
        output2 = self.output_layer2(output)
        return output2, output


parser = argparse.ArgumentParser()
parser.add_argument('channel', type=int)
args = parser.parse_args()
channel = args.channel

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
learning_rate = 0.001
num_epochs = 30
input_dim = channel
hidden_dim = 150
output_dim = 6
device = 'cpu'


# Initialize model, loss function, and optimizer
model = AttentionModel(input_dim=channel, hidden_dim=hidden_dim, num_layers=2, output_dim=128)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


best_train_loss = float('inf')
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size].to(device)
        targets = y_train[i:i+batch_size].to(device)

        outputs, _ = model(inputs)
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
        epoch_acc += accuracy_score(predicted.cpu().numpy(), targets.cpu().numpy()) * inputs.shape[0]

    epoch_loss /= len(X_train) / batch_size
    epoch_acc /= len(y_train)


    print(f'Epoch {epoch+1}, loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}')



model.load_state_dict(best_model_state)

_, x_test = model(X_test)
x_test = x_test.detach().numpy()
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Run PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_test)

# Get unique labels and their corresponding colors
unique_labels = np.unique(y_test)
colors = plt.cm.get_cmap('tab10', len(unique_labels))

# Plot the graph with colored labels
for i, label in enumerate(unique_labels):
    indices = np.where(y_test == label)
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], c=colors(i), label=label)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result with Labels')
plt.legend()
plt.show()
