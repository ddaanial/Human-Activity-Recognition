import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from pytorch_metric_learning import losses
from sklearn.manifold import TSNE

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.01):
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
        return losses.NTXentLoss(temperature=0.01)(logits, torch.squeeze(labels))

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
num_epochs = 40
batch_size = 32

# Define the convolutional neural network model
class ConvNet(nn.Module):
    def __init__(self, input_ch):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 36, 128)
        #self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 36)
        x = torch.relu(self.fc1(x))
        #x = self.fc2(x)
        return x

# Initialize the model
model = ConvNet(input_ch=channel)

# Define the loss function and optimizer
criterion = SupervisedContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_train_loss = float('inf')
# Train the model
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        # Get a batch of data
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)
        #print(outputs.shape, labels.shape)
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
    # Print the training loss after each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.load_state_dict(best_model_state)

new_logits = []
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        # Get a batch of data
        inputs = X_test[i:i+batch_size]
        logits = model(inputs)
        new_logits.extend(logits.cpu().numpy())

data = np.array(new_logits)



tsne = TSNE(n_components=2, random_state=123)
X_tsne = tsne.fit_transform(data)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test)
plt.title('t-SNE Plot of Contrastive')
plt.savefig('tsne_contrastive.png')


