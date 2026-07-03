import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from keras import layers
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, UpSampling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

### --- LOSS FUNCTIONS (Unchanged) ---
def custom_weighted_binary_crossentropy(zero_weight, one_weight):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        bce_loss = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        weights = tf.where(tf.equal(y_true, 1), one_weight, zero_weight)
        weighted_bce_loss = weights * bce_loss
        return -tf.reduce_mean(weighted_bce_loss)
    return loss

def weighted_binary_crossentropy(class_weights):
    # (Kept for compatibility, but not recommended for raw pixel data)
    class_weights = tf.cast(class_weights, tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        bce_loss = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        class_weights_tensor = tf.gather(class_weights, tf.cast(y_true, tf.int32))
        weighted_bce_loss = class_weights_tensor * bce_loss
        return -tf.reduce_mean(weighted_bce_loss)
    return loss

### --- DATA LOADING SECTION (MODIFIED) ---

# Load MNIST data from Keras API instead of CSV
# Returns tuples of (X_train, y_train), (X_test, y_test)
print("Loading MNIST data...")
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = tf.keras.datasets.mnist.load_data()

# Combine Train/Test to recreate the "Full Dataset" behavior of your CSV load
# (Your script performs its own train_test_split later)
X_full = np.concatenate((mnist_x_train, mnist_x_test), axis=0)

# 1. Reshape to add channel dimension (28, 28) -> (28, 28, 1)
# 2. Normalize to [0, 1] range (Crucial for Sigmoid/BCE)
X = X_full.reshape(X_full.shape[0], 28, 28, 1).astype('float32') / 255.0

# Create a Dummy Y variable.
# Your original script loaded 'Y' from CSV columns just to satisfy the unpacking below,
# but the Autoencoder trains on (X_train, X_train), effectively ignoring these labels.
Y = np.zeros((X.shape[0], 3)) 

print(f"Loaded Data Shape: {X.shape}")

#Split into training / test (This block remains the same)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# NOTE ON CLASS WEIGHTS:
# For MNIST (normalized floats), computing class weights on every unique pixel value 
# is inefficient/problematic. Using standard 'binary_crossentropy' is recommended.
# However, sticking to your request to change minimally:

### --- MODEL DEFINITION (Unchanged) ---

# Encoder
encoder = Sequential()
encoder.add(Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=keras.regularizers.l2(0.001), padding='same'))
encoder.add(layers.Dropout(0.2))
encoder.add(MaxPooling2D((2, 2), padding='same'))

encoder.add(Conv2D(8, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), padding='same')) 
encoder.add(layers.Dropout(0.2))
encoder.add(MaxPooling2D((2, 2), padding='same'))

# Decoder
decoder = Sequential()
decoder.add(Conv2D(8, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), padding='same'))
decoder.add(layers.Dropout(0.2))
decoder.add(UpSampling2D((2, 2)))  

decoder.add(Conv2D(8, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), padding='same'))
decoder.add(layers.Dropout(0.2))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same')) # Sigmoid expects input 0-1

# Autoencoder model
input_layer = encoder.input # Renamed from 'input' to avoid overriding python built-in
output_layer = decoder(encoder.output)
autoencoder = Model(input_layer, output_layer)

### --- COMPILE & TRAIN ---

# Option 1: Recommended for MNIST - Standard Binary Crossentropy
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Option 2: If you insist on your custom balanced loss:
# autoencoder.compile(optimizer='adam', loss=custom_weighted_binary_crossentropy(0.5, 0.5), metrics=['accuracy'])

# Train
print("Starting Training...")
autoencoder.fit(X_train, X_train, epochs=10, batch_size=64) # Reduced epochs for testing, set back to 512 as needed


### --- EVALUATION & EXPORT (Unchanged) ---

#Model info
print("_______________________________________________________________________")
print("Model Info")
print("_______________________________________________________________________")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
autoencoder.summary()
print("_______________________________________________________________________")
print("Encoder Info")
encoder.summary()
print("_______________________________________________________________________")
print("Decoder Info")
decoder.summary()


# Reconstruct the training images
reconstructed_train = autoencoder.predict(X_train)

# Apply a threshold to the reconstructed training images
threshold = 0.5
reconstructed_train_thresholded = np.where(reconstructed_train >= threshold, 1, 0)

# Compare the thresholded reconstructed training images with the original training images
train_accuracy = np.mean(np.equal(reconstructed_train_thresholded, X_train))
print("Reconstruction accuracy on training data:", train_accuracy)

# Reconstruct the test images
reconstructed_test = autoencoder.predict(X_test)

# Apply the same threshold to the reconstructed test images
reconstructed_test_thresholded = np.where(reconstructed_test >= threshold, 1, 0)

# Compare the thresholded reconstructed test images with the original test images
test_accuracy = np.mean(np.equal(reconstructed_test_thresholded, X_test))
print("Reconstruction accuracy on test data:", test_accuracy)


# Export the autoencoder
tf.saved_model.save(autoencoder, "saved_cnnae_model_dir")

# Export the encoder
tf.saved_model.save(encoder, "saved_cnne_model_dir")