import tensorflow as tf
from tensorflow.keras import layers, Model
from datasets import load_dataset
from transformers import TFAutoModel, AutoTokenizer
from rouge_score import rouge_scorer

# =====================
# 1. Agent Encoder
# =====================
class AgentEncoder(tf.keras.Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bi_gru = layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True))

    def call(self, x):
        return self.bi_gru(x)

# =====================
# 2. Communication Layer
# =====================
class CommunicationLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_agents):
        super().__init__()
        self.dense = layers.Dense(hidden_dim * 2)
        self.num_agents = num_agents

    def call(self, agent_states):
        messages = []
        for i in range(self.num_agents):
            others = [agent_states[j] for j in range(self.num_agents) if j != i]
            mean_msg = tf.reduce_mean(tf.stack(others, axis=0), axis=0)
            messages.append(self.dense(mean_msg))
        return [agent_states[i] + messages[i] for i in range(self.num_agents)]

# =====================
# 3. Decoder (simplified)
# =====================
class SimpleDecoder(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.gru = layers.GRU(hidden_dim, return_sequences=True)
        self.output_layer = layers.Dense(output_dim)

    def call(self, encoder_outputs):
        x = self.gru(encoder_outputs)
        return self.output_layer(x)

# =====================
# 4. Full DCA Model
# =====================
class DCAModel(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim, num_agents, comm_rounds):
        super().__init__()
        self.num_agents = num_agents
        self.comm_rounds = comm_rounds

        self.encoders = [AgentEncoder(hidden_dim) for _ in range(num_agents)]
        self.comm_layer = CommunicationLayer(hidden_dim, num_agents)
        self.decoder = SimpleDecoder(hidden_dim, output_dim)

    def call(self, inputs):
        agent_states = [self.encoders[i](inputs[i]) for i in range(self.num_agents)]
        for _ in range(self.comm_rounds):
            agent_states = self.comm_layer(agent_states)
        combined = tf.reduce_mean(tf.stack(agent_states, axis=0), axis=0)
        return self.decoder(combined)

# =====================
# 5. Load CNN/DailyMail Dataset
# =====================
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"].select(range(500))

# =====================
# 6. Tokenizer and Preprocessing
# =====================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFAutoModel.from_pretrained("bert-base-uncased")

num_agents = 3
max_input_length = 512


def split_into_chunks(text, num_chunks=3):
    sentences = text.split(". ")
    chunk_size = max(1, len(sentences) // num_chunks)
    return [". ".join(sentences[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_chunks)]


def embed_article_chunks(article):
    chunks = split_into_chunks(article, num_chunks=num_agents)
    embedded_chunks = []
    for chunk in chunks:
        tokens = tokenizer(chunk, padding="max_length", truncation=True, max_length=max_input_length, return_tensors="tf")
        embedding = bert_model(tokens.input_ids).last_hidden_state
        embedded_chunks.append(embedding)
    return embedded_chunks

# =====================
# 7. Training Setup
# =====================
hidden_dim = 64
output_dim = tokenizer.vocab_size
comm_rounds = 2

model = DCAModel(hidden_dim, output_dim, num_agents, comm_rounds)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# =====================
# 8. Training Loop (1 Epoch Demo)
# =====================
for example in train_data:
    embedded_inputs = embed_article_chunks(example["article"])
    labels = tokenizer(example["highlights"], padding="max_length", truncation=True, max_length=128, return_tensors="tf").input_ids












    with tf.GradientTape() as tape:


        predictions = model(embedded_inputs)  # (batch_size, seq_len, vocab_size)

        # Flatten predictions to (batch_size * seq_len, vocab_size)
        predictions = tf.reshape(predictions, [-1, output_dim])

        # Tokenize and flatten labels to (batch_size * seq_len,)
        labels = tokenizer(example["highlights"], padding="max_length", return_tensors="tf").input_ids
        labels = tf.reshape(labels, [-1])

        loss = loss_fn(labels, predictions)


    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("Loss:", loss.numpy())
