import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

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

# Calculate mean and variance along time and axis dimensions
mean_data = np.mean(X, axis=2)  
var_data = np.var(X, axis=2)    

# Reshape mean and variance arrays into a single array with 6 features
X = np.concatenate((mean_data, var_data), axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Fit a Random Forest model on the training data
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Fit an SVM model on the training data
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Fit a Logistic Regression model on the training data
lr = LogisticRegression(max_iter= 1000)
lr.fit(X_train, y_train)

# Fit a K-Nearest Neighbors model on the training data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Fit a Decision Tree model on the training data
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Fit a Gradient Boosting model on the training data
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)

models = [rf, svm, lr, knn, dt, gbm]
models_dict = {
    "RandomForest": rf,
    "LogisticRegression": lr,
    "SVM": svm,
    "KNN": knn,
    "decision_tree": dt,
     "grad_boosting": gbm
}

if channel == 3:
    labels = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
else:
    labels = ['Walk', 'Bike', 'Upstairs', 'Downstairs', 'Run', 'bus/taxi']

# Evaluation
for model_name, model in models_dict.items():
    # make predictions on test set
    predicted = model.predict(X_test)
    acc = accuracy_score(y_test, predicted)
    f1 =f1_score(y_test, predicted, average='weighted')
    if channel == 3:
        with open(f'Results/ML/{model_name}/WISDM/accuracy.txt', 'w') as file:
            # Write the accuracy and loss to the file
            file.write(f'test accuracy: {acc}\n')
            file.write(f'F1 score: {f1}')
    else:
        with open(f'Results/ML/{model_name}/Meta_Har/accuracy.txt', 'w') as file:
            # Write the accuracy and loss to the file
            file.write(f'test accuracy: {acc}\n')
            file.write(f'F1 score: {f1}')

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
        fig.savefig(f"Results/ML/{model_name}/WISDM/{model_name}_confusion_matrix.png")
    else:
        fig.savefig(f"Results/ML/{model_name}/Meta_Har/{model_name}_confusion_matrix.png")

