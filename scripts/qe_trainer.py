# ---------------------------------------------------------
# SQLITE FIX: Must be at the TOP of the script
# ---------------------------------------------------------
import sys
try:
    # Try to load the modern binary we installed
    __import__('pysqlite3')
    # Replace the standard sqlite3 module with the new one
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If pysqlite3-binary is not installed, print an error
    print("ERROR: pysqlite3-binary not installed! Run 'pip install pysqlite3-binary'")
    exit()

# ---------------------------------------------------------
# NOW continue with your normal imports...
# ---------------------------------------------------------

import os
import sys
import ast 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import chromadb


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CHROMA_DB_PATH = "./chroma_db_mnist"
COLLECTION_NAME = "mnist_sparse_collection"

ENCODER_PATH = "./saved_cnne_model_dir"
VALUE_ENC_PATH = "./saved_mnist_classifier_dir" 

SAVE_PATH_RESIDUAL_CNN = "./saved_residual_cnn_with_VE_guidance"
EMBEDDING_DIM = 392 # Must match your Frozen Encoder output size
NUM_NEIGHBORS = 5       
BATCH_SIZE = 64
EPOCHS = 10             
LEARNING_RATE = 0.001


# ---------------------------------------------------------
# 1. DATA PREPARATION (MNIST)
# ---------------------------------------------------------
print("_______________________________________________________________________")
print("Loading Data")
print("_______________________________________________________________________")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X_full = np.concatenate((x_train, x_test), axis=0)
Y_full = np.concatenate((y_train, y_test), axis=0)

X_processed = X_full.reshape(X_full.shape[0], 28, 28, 1).astype('float32') / 255.0
Y_onehot = tf.keras.utils.to_categorical(Y_full, 10)

indices = np.arange(len(X_processed))
idx_train, idx_test, _, _ = train_test_split(indices, Y_full, test_size=0.2, stratify=Y_full)

X_tr = X_processed[idx_train]; y_tr_int = Y_full[idx_train]; y_tr_hot = Y_onehot[idx_train]
X_te = X_processed[idx_test]; y_te_int = Y_full[idx_test]; y_te_hot = Y_onehot[idx_test]


# ---------------------------------------------------------
# 2. LOAD CHROMADB MEMORY BANK
# ---------------------------------------------------------
print("\nConnecting to Vector Database...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

results = collection.get(include=['embeddings', 'metadatas'])
db_vecs_raw = np.array(results['embeddings']).astype('float32')

db_labels_raw = []
for m in results['metadatas']:
    try: db_labels_raw.append(ast.literal_eval(m['one_hot_vector']))
    except: db_labels_raw.append([0]*10) 
        
db_labels_raw = np.array(db_labels_raw).astype('float32')
MEM_BANK_VECS = tf.constant(db_vecs_raw)
MEM_BANK_LABELS = tf.constant(db_labels_raw)


# ---------------------------------------------------------
# 3. ARCHITECTURE DEFINITIONS (UPDATED)
# ---------------------------------------------------------

class FrozenEncoderLayer(layers.Layer):
    """Takes Image -> Flattened Latent Z"""
    def __init__(self, module, **kwargs):
        super(FrozenEncoderLayer, self).__init__(**kwargs)
        self.module = module
        self.trainable = False
        
    def call(self, inputs):
        res = self.module(inputs)
        return tf.reshape(res, [tf.shape(res)[0], -1]) 

class ResidualCNN(keras.Model):
    """
    NEW: Takes RAW IMAGE -> Outputs Adjustment Vector (Same Size as Z)
    Uses convolutions to extract local spatial patterns lost by the base AE.
    """
    def __init__(self, target_dim):
        super().__init__()
        
        self.conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        self.flatten = layers.Flatten()
        
        # Intermediate dense layer to project features down
        self.dense_proj = layers.Dense(128, activation='relu')
        
        # Final output layer MUST match target_dim (Embedding Dim)
        # Activation is LINEAR so we can add positive/negative corrections
        self.out_layer = layers.Dense(target_dim, activation='linear') 

    def call(self, raw_image_inputs):
        x = self.conv1(raw_image_inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense_proj(x)
        delta_z = self.out_layer(x)
        return delta_z

class AttentionRetriever(Model):
    """Fuses Base Encoder + Residual CNN, then Queries DB"""
    def __init__(self, enc, cnn_residual):
        super().__init__()
        self.enc = enc
        self.cnn_residual = cnn_residual
        
    def call(self, inputs):
        # Branch 1: Standard Encoding (Image -> Z_base)
        z_base = self.enc(inputs)
        
        # Branch 2: Residual Correction (Image -> delta_Z)
        # The CNN looks at the ORIGINAL INPUT again!
        delta = self.cnn_residual(inputs)
        
        # FUSE: Add the correction vector to the base embedding
        z_adjusted = z_base + delta
        
        # Normalize for Cosine Search
        z_norm = tf.linalg.l2_normalize(z_adjusted, axis=1)
        
        sim_matrix = tf.matmul(z_norm, MEM_BANK_VECS, transpose_b=True)
        
        # Get Top-K Neighbors
        values, indices = tf.math.top_k(sim_matrix, k=NUM_NEIGHBORS)
        attn_weights = tf.nn.softmax(values, axis=1)
        
        # Retrieve Labels
        neighbor_labels = tf.gather(MEM_BANK_LABELS, indices) 
        
        # Weighted Prediction
        pred_retrieval = tf.reduce_sum(
            tf.expand_dims(attn_weights, -1) * neighbor_labels, 
            axis=1
        )
        return pred_retrieval

class GuidedSystem(Model):
    """
    Orchestrates:
    1. Retrieval Branch (Trains the CNN Residual)
    2. Value Encoder Branch (Frozen Classifier)
    3. Fusion (Average)
    """
    def __init__(self, retriever, value_encoder_path):
        super().__init__()
        self.retriever = retriever
        
        print(f"Loading Value Encoder from {value_encoder_path}...")
        self.value_encoder = models.load_model(value_encoder_path)
        self.value_encoder.trainable = False 
        print("Value Encoder Loaded & Frozen.")
        
    def call(self, inputs):
        # 1. Retrieval Output (Depends on trainable CNN Residual)
        pred_ret = self.retriever(inputs)
        
        # 2. Value Encoder Output (Fixed Reference)
        pred_ve = tf.stop_gradient(self.value_encoder(inputs))
        
        # 3. Average Fusion
        final_pred = (pred_ret + pred_ve) / 2.0
        
        return final_pred


# ---------------------------------------------------------
# 4. INSTANTIATION & SETUP
# ---------------------------------------------------------

# Load Base AE Encoder
loaded_obj = tf.saved_model.load(ENCODER_PATH)
frozen_enc_layer = FrozenEncoderLayer(loaded_obj)

# Create new CNN Residual Component
trainable_cnn = ResidualCNN(target_dim=EMBEDDING_DIM)

# Build Retriever System
retriever_branch = AttentionRetriever(frozen_enc_layer, trainable_cnn)

# Create Full System
system_model = GuidedSystem(retriever_branch, VALUE_ENC_PATH)


optimizer = Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------

print("\n_______________________________________________________________________")
print("Starting Training: Optimizing CNN Residual via Value Encoder Guidance")
print("_______________________________________________________________________")

dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr_hot)).shuffle(10000).batch(BATCH_SIZE)
history_loss = []
history_acc = []

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    for step, (x_batch, y_true) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred_prob = system_model(x_batch, training=True)
            
            loss_val = loss_fn(y_true, y_pred_prob)
            
        # Gradients will ONLY update 'trainable_cnn' weights automatically
        grads = tape.gradient(loss_val, system_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, system_model.trainable_weights))
        
        epoch_loss_avg.update_state(loss_val)
        epoch_acc_metric.update_state(y_true, y_pred_prob)
        
        if step % 50 == 0:
             print(f"Step {step}: Loss = {loss_val.numpy():.4f}, Acc = {epoch_acc_metric.result().numpy():.4f}")
            
    history_loss.append(epoch_loss_avg.result().numpy())
    history_acc.append(epoch_acc_metric.result().numpy())
    
    print(f">>> End Epoch: Avg Loss = {history_loss[-1]:.4f}, Avg Acc = {history_acc[-1]:.4f}")


# ---------------------------------------------------------
# 6. EVALUATION & EXPORT
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Evaluation Results")
print("_______________________________________________________________________")

print("Calculating Test Set Metrics...")

y_ve_logits = system_model.value_encoder.predict(X_te)
y_ve_cls = np.argmax(y_ve_logits, axis=1)

y_final_probs = system_model.predict(X_te)
y_final_cls = np.argmax(y_final_probs, axis=1)

acc_ve = accuracy_score(y_te_int, y_ve_cls)
acc_final = accuracy_score(y_te_int, y_final_cls)

print(f"Value Encoder (Baseline) Accuracy : {acc_ve:.4f}")
print(f"CNN-Ensemble Accuracy           : {acc_final:.4f}")

if acc_final > acc_ve:
    print(f"(The Spatial CNN improved results!)")

print("\nClassification Report:")
print(classification_report(y_te_int, y_final_cls))

# Save the trained component
print("\nSaving Trained Residual CNN...")
tf.saved_model.save(trainable_cnn, SAVE_PATH_RESIDUAL_CNN)
print(f"Saved to: {SAVE_PATH_RESIDUAL_CNN}")