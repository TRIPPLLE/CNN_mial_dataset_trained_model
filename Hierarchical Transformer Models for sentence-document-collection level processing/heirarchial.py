import nltk
nltk.download('punkt')

from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import tensorflow as tf
from spektral.layers import GCNConv
from spektral.utils import normalized_adjacency
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset (multiple docs → we simulate 2 articles from validation set)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:2]")

# Sentence tokenizer and embedding model
tokenizer = sent_tokenize
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Encode sentences from each document
doc_embeddings = []
all_sentences = []
doc_id = []

for i, sample in enumerate(dataset):
    sentences = tokenizer(sample['article'])
    sentence_embeddings = encoder.encode(sentences)
    
    all_sentences.extend(sentences)
    doc_embeddings.extend(sentence_embeddings)
    doc_id.extend([i] * len(sentences))  # track which doc it came from

X = np.array(doc_embeddings)
print(f"🧠 Total sentences: {len(X)} from {len(dataset)} documents.")

# Step 2: Build hierarchical graph (sentence nodes, intra/inter-doc edges)
G = nx.Graph()
for i in range(len(X)):
    G.add_node(i)

# Add intra-document edges (cosine similarity)
cos_sim = cosine_similarity(X)
threshold_intra = 0.7
for i in range(len(X)):
    for j in range(i + 1, len(X)):
        if doc_id[i] == doc_id[j] and cos_sim[i][j] > threshold_intra:
            G.add_edge(i, j, weight=cos_sim[i][j])

# Add inter-document edges (weak communication links)
threshold_inter = 0.85
for i in range(len(X)):
    for j in range(i + 1, len(X)):
        if doc_id[i] != doc_id[j] and cos_sim[i][j] > threshold_inter:
            G.add_edge(i, j, weight=cos_sim[i][j])

# Convert to tensors
adj = nx.to_numpy_array(G)
A = normalized_adjacency(adj)
X = tf.convert_to_tensor(X, dtype=tf.float32)
A = tf.convert_to_tensor(A, dtype=tf.float32)

# Step 3: GCN Model (Communicating Agent)
class HCA_GCN(Model):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(128, activation='relu')
        self.dense = Dense(1)

    def call(self, inputs):
        x, a = inputs
        x = self.gcn1([x, a])
        out = self.dense(x)
        return tf.squeeze(out, axis=-1)

model = HCA_GCN()
model.compile(optimizer=Adam(1e-3), loss='mse')

# Unsupervised: dummy targets
y_dummy = tf.ones(len(X))
model.fit([X, A], y_dummy, epochs=10, verbose=0)

# Predict importance scores for all sentences
scores = model([X, A]).numpy()
top_k = 5
top_indices = np.argsort(scores)[-top_k:][::-1]

# Output summary
print("\n🧠 HCA-GNN Summary:")
for idx in top_indices:
    print("•", all_sentences[idx])

# Optional: Print reference summaries
print("\n📚 Reference Summary 1:\n", dataset[0]['highlights'])
print("\n📚 Reference Summary 2:\n", dataset[1]['highlights'])
