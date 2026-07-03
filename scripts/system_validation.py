# ---------------------------------------------------------
# SQLITE FIX
# ---------------------------------------------------------
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("ERROR: pysqlite3-binary not installed!")
    exit()

# ---------------------------------------------------------
# IMPORTS (FIXED)
# ---------------------------------------------------------
import os
import ast
import time # <--- This was missing!
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import chromadb

# ... (Rest of the script remains exactly the same) ...

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Paths MUST match where you saved everything previously
ENCODER_PATH = "./saved_cnne_model_dir"
VALUE_ENC_PATH = "./saved_mnist_classifier_dir"
TRAINED_CNN_PATH = "./saved_residual_cnn_with_VE_guidance"

CHROMA_DB_PATH = "./chroma_db_mnist"
COLLECTION_NAME = "mnist_sparse_collection"

EMBEDDING_DIM = 392 
NUM_NEIGHBORS = 5


# ---------------------------------------------------------
# 1. ARCHITECTURE BLUEPRINTS (Must redefine to house the loaded modules)
# ---------------------------------------------------------

class FrozenEncoderLayer(layers.Layer):
    def __init__(self, module, **kwargs):
        super(FrozenEncoderLayer, self).__init__(**kwargs)
        self.module = module
        self.trainable = False
        
    def call(self, inputs):
        res = self.module(inputs)
        return tf.reshape(res, [tf.shape(res)[0], -1]) 

class ResidualCNN(keras.Model):
    # Skeleton class definition isn't strictly needed if we load directly,
    # but helpful for type safety if we wanted to inspect summaries.
    # We will mostly rely on the loaded object having .call() functionality.
    def __init__(self):
        super().__init__()
    
    # This will be overwritten by the loaded weights

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
        
        # ---------------------------------------------------------
        # THE FIX: Explicitly Cast to Float32
        # Prevents "Input y has type int32" Error
        # ---------------------------------------------------------
        neighbor_labels = tf.cast(neighbor_labels, tf.float32)

        # Weighted Prediction
        pred_retrieval = tf.reduce_sum(
            tf.expand_dims(attn_weights, -1) * neighbor_labels, 
            axis=1
        )
        return pred_retrieval

class GuidedSystem(Model):
    """
    Final Assembly: Retrieval + Value Encoder (Average Fusion)
    """
    def __init__(self, retriever, value_encoder_path):
        super().__init__()
        self.retriever = retriever
        
        print("Loading Value Encoder...")
        self.value_encoder = models.load_model(value_encoder_path)
        self.value_encoder.trainable = False
        print("VE Loaded.")
        
    def call(self, inputs):
        pred_ret = self.retriever(inputs)
        pred_ve = tf.stop_gradient(self.value_encoder(inputs))
        
        final_pred = (pred_ret + pred_ve) / 2.0
        return final_pred


# ---------------------------------------------------------
# 2. INITIALIZE MEMORY BANK (Required for Retrieval Math)
# ---------------------------------------------------------
print("Initializing Spatial Hash (Memory Bank)...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

results = collection.get(include=['embeddings', 'metadatas'])
db_vecs_raw = np.array(results['embeddings']).astype('float32')

db_labels_raw = []
for m in results['metadatas']:
    try: db_labels_raw.append(ast.literal_eval(m['one_hot_vector']))
    except: db_labels_raw.append([0]*10) 

# These Global Constants are required by AttentionRetriever.call()
MEM_BANK_VECS = tf.constant(db_vecs_raw)
MEM_BANK_LABELS = tf.constant(db_labels_raw)

print(f"Memory Bank Loaded: {MEM_BANK_VECS.shape[0]} items")


# ---------------------------------------------------------
# 3. LOAD MODELS
# ---------------------------------------------------------

# A. Load Base Encoder (Feature Extractor)
print("\nLoading Feature Encoder (Frozen)...")
loaded_obj = tf.saved_model.load(ENCODER_PATH)
frozen_enc_layer = FrozenEncoderLayer(loaded_obj)

# B. Load Trained CNN Residual (The "Neural Wrapper")
print(f"Loading Trained Residual CNN from {TRAINED_CNN_PATH}...")
if os.path.exists(TRAINED_CNN_PATH):
    trained_cnn = tf.saved_model.load(TRAINED_CNN_PATH)
else:
    raise FileNotFoundError(f"Trained CNN not found at {TRAINED_CNN_PATH}")

# C. Build System
print("Assembling System...")
final_system = GuidedSystem(
    retriever=AttentionRetriever(frozen_enc_layer, trained_cnn),
    value_encoder_path=VALUE_ENC_PATH
)


# ---------------------------------------------------------
# 4. RUN VERIFICATION ON TEST DATA
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("VERIFICATION PHASE")
print("_______________________________________________________________________")

# Load Test Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X_te = x_test.astype('float32') / 255.0
y_te_int = y_test
X_te = np.expand_dims(X_te, -1) # Add channel dim

print(f"Running inference on {len(X_te)} images...")

# Generate Predictions
# Note: This triggers the full pipeline:
# Image -> Enc/Z + CNN/Delta -> Adjusted_Z -> DB_Search(Attn) -> Ret_Pred
# -> VE_Pred -> Average
start_time = time.time()
y_pred_probs = final_system.predict(X_te, batch_size=64, verbose=1)
end_time = time.time()
y_cls_final = np.argmax(y_pred_probs, axis=1)


# ---------------------------------------------------------
# 5. METRICS REPORTING
# ---------------------------------------------------------

acc_overall = accuracy_score(y_te_int, y_cls_final)

print(f"\n{'='*60}")
print(f"FINAL SYSTEM VERIFICATION RESULTS")
print(f"{'='*60}")
print(f"Inference Time: {end_time - start_time:.2f}s")
print(f"System Accuracy: {acc_overall:.4f} ({acc_overall*100:.2f}%)")

print(f"\nDetailed Classification Report:")
print(classification_report(y_te_int, y_cls_final))

# Optional: Confusion Matrix (Raw numbers)
print("Confusion Matrix Snippet:")
print(confusion_matrix(y_te_int, y_cls_final)[:5]) # Print top 5 rows to save space

# Check Specific Components (Sanity Checks)
print("-"*40)
print("COMPONENT SANITY CHECKS:")

# Check 1: Does the system actually use both components?
# We check by comparing pure VE vs Pure Retrieval vs Hybrid
y_ve_only_logits = final_system.value_encoder.predict(X_te, verbose=0)
y_cls_ve = np.argmax(y_ve_only_logits, axis=1)
acc_ve_only = accuracy_score(y_te_int, y_cls_ve)

print(f"1. Value Encoder Standalone Acc : {acc_ve_only:.4f}")
print(f"2. Full Ensemble System Acc      : {acc_overall:.4f}")

if acc_overall >= acc_ve_only:
    diff = ((acc_overall - acc_ve_only) / acc_ve_only) * 100
    print(f"   >>> ENSEMBLE BOOST: +{diff:.2f}% <<<")
else:
    print("   (Note: Ensemble did not improve over standalone. Check Thresholds/Data)")

print("-"*40)
print("Verification Complete.")