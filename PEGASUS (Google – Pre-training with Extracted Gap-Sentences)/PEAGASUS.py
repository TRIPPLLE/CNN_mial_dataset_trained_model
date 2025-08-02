from datasets import load_dataset
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TrainingArguments, Trainer
import evaluate
import torch

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Load tokenizer and model
model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Preprocessing
def preprocess_function(examples):
    inputs = tokenizer(examples["article"], max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments (Updated: epochs = 2)
training_args = TrainingArguments(
    output_dir="./results_pegasus",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,  # ✅ Updated from 3 → 2
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(5000)),
    eval_dataset=tokenized_dataset["validation"].select(range(500)),
)

# Training
trainer.train()

# Evaluation
rouge = evaluate.load("rouge")

def generate_summary(batch):
    inputs = tokenizer(batch["article"], return_tensors="pt", max_length=1024, truncation=True).to("cuda")
    summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=5)
    batch["predicted"] = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return batch

# Run evaluation
model.to("cuda")
sample_data = test_data.select(range(100)).map(generate_summary, batched=False)
results = rouge.compute(predictions=sample_data["predicted"], references=sample_data["highlights"])
print(results)
