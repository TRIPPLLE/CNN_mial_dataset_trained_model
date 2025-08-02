import nltk
nltk.download("punkt")

from datasets import load_dataset
from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"]
val_data = dataset["validation"]

# Load tokenizer and model
model_name = "allenai/led-base-16384"
tokenizer = LEDTokenizer.from_pretrained(model_name)
model = LEDForConditionalGeneration.from_pretrained(model_name)

# Preprocessing
max_input_len = 4096
max_target_len = 256

def preprocess(example):
    # Tokenize inputs
    inputs = tokenizer(
        example["article"],
        max_length=max_input_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    targets = tokenizer(
        example["highlights"],
        max_length=max_target_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    inputs["labels"] = targets["input_ids"]
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "labels": targets["input_ids"][0],
        "global_attention_mask": [1] + [0] * (max_input_len - 1)  # First token global
    }

# Tokenize
tokenized_train = train_data.map(preprocess, batched=False, remove_columns=train_data.column_names)
tokenized_val = val_data.map(preprocess, batched=False, remove_columns=val_data.column_names)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./led-cnn-checkpoints",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=1,  # LED is memory heavy!
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save
model.save_pretrained("./led-cnn-final")
tokenizer.save_pretrained("./led-cnn-final")
