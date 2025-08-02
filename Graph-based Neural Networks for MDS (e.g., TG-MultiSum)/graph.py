from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from evaluate import load

# Load article and summary
dataset = load_dataset("cnn_dailymail", "3.0.0")
article = dataset["test"][0]["article"]
reference_summary = dataset["test"][0]["highlights"]

# Split manually without nltk
sentences = article.split(". ")
sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

# Embed and rank
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

G = nx.Graph()
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        if sim > 0.6:
            G.add_edge(i, j, weight=sim)

scores = nx.pagerank(G, weight="weight")
ranked = sorted(((score, sent) for sent, score in zip(sentences, scores.values())), reverse=True)
summary = " ".join([s for _, s in ranked[:5]])

# ROUGE Eval
rouge = load("rouge")
results = rouge.compute(predictions=[summary], references=[reference_summary])

print("📊 ROUGE-1:", results["rouge1"])
print("📊 ROUGE-2:", results["rouge2"])
print("📊 ROUGE-L:", results["rougeL"])
