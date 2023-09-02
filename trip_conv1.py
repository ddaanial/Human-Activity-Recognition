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

X_ = X.numpy()
y_ = y.numpy()

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
#learning_rate = 0.001
#num_epochs = num_epoch
batch_size = 32
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=123)

# Define the data loader for training
train_dataset = CustomDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the data loader for testing
test_dataset = CustomDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the convolutional neural network model
class ConvNet(nn.Module):
    def __init__(self, input_ch):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 36, 128)


    def forward(self,x, x1, x2):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 36)
        x = torch.relu(self.fc1(x))

        x1 = self.pool(torch.relu(self.conv1(x1)))
        x1 = self.pool(torch.relu(self.conv2(x1)))
        x1 = x1.view(-1, 64 * 36)
        x1 = torch.relu(self.fc1(x1))

        x2 = self.pool(torch.relu(self.conv1(x2)))
        x2 = self.pool(torch.relu(self.conv2(x2)))
        x2 = x2.view(-1, 64 * 36)
        x2 = torch.relu(self.fc1(x2))

        return x, x1, x2

# Initialize the model
net = ConvNet(input_ch=channel)
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

        # Print the loss every 100 batches
        # if (i+1) % 100 == 0:
        #     print("Epoch {} | Batch {} | Loss: {:.4f}".format(epoch+1, i+1, loss.item()))

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
#     print(f'Test Accuracy: {accuracy:.2f}%')

#     # Open a file for writing
#     if channel == 3:
#         with open('Results/Triplet/WISDM/Contrastive/WISDM_Result.txt', 'w') as file:
#             # Write the accuracy and F1 score to the file
#             file.write(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
#     else:
#         with open('Results/Triplet/Meta_Har/Contrastive/Meta_Har_Result.txt', 'w') as file:
#             # Write the accuracy and F1 score to the file
#             file.write(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')

# # Save the best model state to a file
# if channel == 3:
#     torch.save(best_model_state, 'Results/Triplet/WISDM/Contrastive/WISDM_best_model.pth')
# else:
#     torch.save(best_model_state, 'Results/Triplet/Meta_Har/Contrastive/Meta_Har_best_model.pth')

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


#     # Print the training loss after each epoch
#     print(f'Epoch {epoch+1}, loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}')

#     # Open a file for writing
#     if channel == 3:
#         with open('Results/Triplet/WISDM/Classifier/WISDM_Loss.txt', 'a') as file:
#             # Write the accuracy 
#             file.write(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}\n')
#     else:
#         with open('Results/Triplet/Meta_Har/Classifier/Meta_Har_Loss.txt', 'a') as file:
#             # Write the accuracy 
#             file.write(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2%}\n')

# # Save the best model state to a file
# if channel == 3:
#     torch.save(best_model_state, 'Results/Triplet/WISDM/Classifier/WISDM_best_model1.pth')
# else:
#     torch.save(best_model_state, 'Results/Triplet/Meta_Har/Classifier/Meta_Har_best_model.pth')


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

#     if channel == 3:
#         with open('Results/Triplet/WISDM/Classifier/WISDM_Result.txt', 'w') as file:
#             # Write the accuracy and loss to the file
#             file.write(f'test loss: {test_loss}\n')
#             file.write(f'test accuracy: {accuracy}\n')
#             file.write(f'F1 Score: {f1}')
#     else:
#         with open('Results/Triplet/Meta_Har/Classifier/Meta_Har_Result.txt', 'w') as file:
#             # Write the accuracy and loss to the file
#             file.write(f'test loss: {test_loss}\n')
#             file.write(f'test accuracy: {accuracy}\n')
#             file.write(f'F1 Score: {f1}')

# print(f'accuracy: {accuracy}') 
# print(f'F1 score: {f1}') 
# print(f'test loss: {test_loss}') 

if channel == 3:
    print(f'Conv1 WISDM Triplet accuracy: {accuracy}')  
else:
    print(f'Conv1 META_HAR Triplet accuracy: {accuracy}')  
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
#     plt.savefig('Results/Triplet/WISDM/WISDM_confusion_matrix.png')
# else:
#     plt.savefig('Results/Triplet/Meta_Har/Meta_Har_confusion_matrix.png')
