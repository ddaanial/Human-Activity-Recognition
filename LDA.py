import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

X_2d = np.array([features_2d.flatten() for features_2d in X.numpy()])
y_ = y.numpy()

# Reduction with LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_2d, y_)

# Plot the result
x1 = X_lda[:,0]
x2 = X_lda[:,1]
plt.scatter(x1, x2, c=y_, edgecolor='none', alpha=.8, cmap=plt.colormaps['viridis'])
plt.xlabel('lda1')
plt.ylabel('lda2')
plt.colorbar()
plt.axis('off')
if channel == 3:
    plt.savefig('WISDM_LDA.png')
else:
    plt.savefig('Meta_Har_LDA.png')

