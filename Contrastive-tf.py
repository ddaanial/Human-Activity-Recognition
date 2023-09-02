
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
    
class Conv1DModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(32, 3, activation="relu", data_format="channels_first")
        self.conv2 = tf.keras.layers.Conv1D(64, 3, activation="relu",  data_format="channels_first")
        self.pool = tf.keras.layers.MaxPool1D(2, data_format="channels_first")
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc1(x)
        return x
        
    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc1(x)
        return x

class Classifier(tf.keras.Model):    

    def __init__(self,
                 n_classes,
                 encoder,
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.encoder = encoder
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(32)
        self.softmax = tf.keras.layers.Dense(n_classes, activation="softmax")

    def call(self, inputs):
        out = self.encoder(inputs)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.softmax(out)
        return out

channel = 6

if channel == 3:
    with open('WISDM.pkl', 'rb') as f:
        X, y = pickle.load(f)
else:
    with open('Meta_Har.pkl', 'rb') as f:
        X, y = pickle.load(f)

X = X.numpy()
y = y.numpy()
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

loss = SupervisedContrastiveLoss(temperature=0.4)
optimizer = tf.keras.optimizers.Adam(0.01)
encoder = Conv1DModel()
encoder.compile(loss = loss, optimizer = optimizer)

##### Comment these three lines to skip contrastive learning
encoder.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 40)
for layer in encoder.layers:
    layer.trainable = False
##############################
        
metrics = ["accuracy"]
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.01)
classifier = Classifier(n_classes=n_classes, encoder = encoder)
classifier.compile(loss = loss, optimizer = optimizer,  metrics = metrics)
classifier.fit(X_train, y_train, validation_data=(X_test, y_test),epochs = 20)

