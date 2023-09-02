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
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        epoch_acc += accuracy_score(predicted.cpu().numpy(), labels.cpu().numpy()) * inputs.shape[0]

    epoch_loss /= len(X_train) / batch_size
    epoch_acc /= len(y_train)
    # Print the training loss after each epoch
    print(f'Epoch {epoch+1}, loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}')

    # Open a file for writing
    if channel == 3:
        with open('Results/Conv1/WISDM/WISDM_Loss.txt', 'a') as file:
            # Write the accuracy
            file.write(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}\n')
    else:
        with open('Results/Conv1/Meta_Har/Meta_Har_Loss.txt', 'a') as file:
            # Write the accuracy 
            file.write(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}\n')

# Save the best model state to a file
if channel == 3:
    torch.save(best_model_state, 'Results/Conv1/WISDM/WISDM_best_model.pth')
else:
    torch.save(best_model_state, 'Results/Conv1/Meta_Har/Meta_Har_best_model.pth')


model.load_state_dict(best_model_state)

# Put the model in evaluation mode
model.eval()
# Evaluate the model on your testing data
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / total
    f1 = f1_score(predicted,y_test,average='weighted')

    # Compute loss
    y_test = y_test.long()
    loss = criterion(outputs,y_test)

    # Open a file for writing
    if channel == 3:
        with open('Results/Conv1/WISDM/WISDM_Result.txt', 'w') as file:
            # Write the accuracy and F1 score to the file
            file.write(f'Accuracy: {accuracy}\n')
            file.write(f'F1 Score: {f1}\n')
            file.write(f'Loss: {loss.item()}')
    else:
        with open('Results/Conv1/Meta_Har/Meta_Har_Result.txt', 'w') as file:
            # Write the accuracy and F1 score to the file
            file.write(f'Accuracy: {accuracy}\n')
            file.write(f'F1 Score: {f1}\n')
            file.write(f'Loss: {loss.item()}')

    print("F1 Score:", f1)
    print('Test Accuracy:', accuracy)
    print("Loss:", loss.item())

if channel == 3:
    labels = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
else:
    labels = ['Walk', 'Bike', 'Upstairs', 'Downstairs', 'Run', 'bus/taxi']

# Compute confusion matrix
cm = confusion_matrix(y_test, predicted)
# normalize confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax.text(j, i, format(cm_norm[i, j], '.2f'),
                ha="center", va="center",
                color="black" if cm_norm[i, j] > cm_norm.max() / 2. else "black")
        
fig.tight_layout()

if channel == 3:
    plt.savefig('Results/Conv1/WISDM/WISDM_confusion_matrix.png')
else:
    plt.savefig('Results/Conv1/Meta_Har/Meta_Har_confusion_matrix.png')
