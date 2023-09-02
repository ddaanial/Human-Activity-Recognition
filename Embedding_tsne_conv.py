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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.manifold import TSNE

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

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Hyperparameter
learning_rate = 0.001
num_epochs = 30
batch_size = 32

# Define the convolutional neural network model
class ConvNet(nn.Module):
    def __init__(self, input_ch):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 36, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 36)
        x1 = torch.relu(self.fc1(x))
        x = self.fc2(x1)
        return x, x1

# Initialize the model
model = ConvNet(input_ch=channel)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_train_loss = float('inf')
# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    for i in range(0, len(X_train), batch_size):
        # Get a batch of data
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]
        labels = labels.long()
        # Forward pass
        outputs, _ = model(inputs)

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

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        epoch_acc += accuracy_score(predicted.cpu().numpy(), labels.cpu().numpy()) * inputs.shape[0]

    epoch_loss /= len(X_train) / batch_size
    epoch_acc /= len(y_train)
    # Print the training loss after each epoch
    print(f'Epoch {epoch+1}, loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}')



model.load_state_dict(best_model_state)

# Put the model in evaluation mode
model.eval()
new_logits = []
# Evaluate the model on your testing data
with torch.no_grad():
    _, outputs = model(X_test)
    new_logits.extend(outputs.cpu().numpy())

data = np.array(new_logits)



tsne = TSNE(n_components=2, random_state=123)
X_tsne = tsne.fit_transform(data)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test)
plt.title('t-SNE Plot of Conv')
plt.savefig(f'conv_{channel}_tsne.png')


