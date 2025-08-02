import nltk
nltk.download("punkt")

from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"]
val_data = dataset["validation"]

# Load tokenizer and model
model_name = "facebook/bart-base"  # or "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Preprocessing function
max_input_len = 512
max_target_len = 128

def preprocess(example):
    inputs = tokenizer(
        example["article"],
        max_length=max_input_len,
        padding="max_length",
        truncation=True
    )
    targets = tokenizer(
        example["highlights"],
        max_length=max_target_len,
        padding="max_length",
        truncation=True
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize
tokenized_train = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names)
tokenized_val = val_data.map(preprocess, batched=True, remove_columns=val_data.column_names)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./bart-cnn-checkpoints",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=500,
    save_steps=10000,
    eval_steps=2000,
    logging_dir="./logs",
    fp16=True  # if using GPU with float16 support
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save final model
model.save_pretrained("./bart-cnn-final")
tokenizer.save_pretrained("./bart-cnn-final")
