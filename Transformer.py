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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch.nn.functional as F
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, hidden_dim, num_layers, dropout):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_1 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(output_dim, 6)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1) # Adjust the dimensions for transformer input
        x = self.encoder(x)
        x = x.transpose(0, 1) # Adjust the dimensions back to (batch_size, seq_len, hidden_dim)
        # Calculate the average along the sequence length dimension
        x_avg = torch.mean(x, dim=1)

        # Apply linear transformation to the averaged output
        x = self.relu(self.fc_1(x_avg))
        output = self.fc_2(x)

        return output




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
hidden_dim = 64
output_dim = 128
device = 'cpu'
num_heads = 8
num_layers = 2
dropout = 0.1


# Initialize model, loss function, and optimizer
model = TransformerModel(input_dim, output_dim, num_heads, hidden_dim, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


best_train_loss = float('inf')
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size].to(device)
        targets = y_train[i:i+batch_size].to(device)

        outputs = model(inputs)

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
    # Open a file for writing
    if channel == 3:
        with open('Results/Transformer/WISDM/WISDM_Loss.txt', 'a') as file:
            # Write the accuracy 
            file.write(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}\n')
    else:
        with open('Results/Transformer/Meta_Har/Meta_Har_Loss.txt', 'a') as file:
            # Write the accuracy 
            file.write(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}\n')

# Save the best model state to a file
if channel == 3:
    torch.save(best_model_state, 'Results/Transformer/WISDM/WISDM_best_model.pth')
else:
    torch.save(best_model_state, 'Results/Transformer/Meta_Har/Meta_Har_best_model.pth')


model.load_state_dict(best_model_state)


preds = []
# Evaluate model on test set
with torch.no_grad():
    model.eval()
    test_loss = 0
    test_acc = 0
    
    for i in range(0, len(X_test), batch_size):
        inputs = X_test[i:i+batch_size].to(device)
        targets = y_test[i:i+batch_size].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_acc += accuracy_score(predicted.cpu().numpy(), targets.cpu().numpy()) * inputs.shape[0]
        preds.extend(list(predicted.cpu().numpy()))

    test_loss /= len(X_test) / batch_size
    test_acc /= len(X_test)

    f1 = f1_score(preds,y_test,average='weighted')

    if channel == 3:
        with open('Results/Transformer/WISDM/WISDM_Result.txt', 'w') as file:
            # Write the accuracy and loss to the file
            file.write(f'test loss: {test_loss}\n')
            file.write(f'test accuracy: {test_acc}\n')
            file.write(f'F1 Score: {f1}\n')
    else:
        with open('Results/Transformer/Meta_Har/Meta_Har_Result.txt', 'w') as file:
            # Write the accuracy and loss to the file
            file.write(f'test loss: {test_loss}\n')
            file.write(f'test accuracy: {test_acc}\n')
            file.write(f'F1 Score: {f1}\n')

print(f'Test loss: {test_loss:.4f}, test accuracy: {test_acc:.2%}, F1 Score: {f1}')

if channel == 3:
    labels = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
else:
    labels = ['Walk', 'Bike', 'Upstairs', 'Downstairs', 'Run', 'bus/taxi']

# Compute confusion matrix
cm = confusion_matrix(y_test, preds)
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
    plt.savefig('Results/Transformer/WISDM/WISDM_confusion_matrix.png')
else:
    plt.savefig('Results/Transformer/Meta_Har/Meta_Har_confusion_matrix.png')

