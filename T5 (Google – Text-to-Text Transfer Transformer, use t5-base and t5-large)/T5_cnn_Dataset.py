from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import nltk
nltk.download("punkt")

# Load dataset (train, validation)
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"]
val_data = dataset["validation"]

# Load tokenizer and model
model_name = "t5-base"  # or t5-small, t5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocessing function
max_input_len = 512
max_target_len = 128

def preprocess(example):
    input_text = "summarize: " + example["article"]
    target_text = example["highlights"]

    input_encoding = tokenizer(
        input_text,
        max_length=max_input_len,
        padding="max_length",
        truncation=True,
    )

    target_encoding = tokenizer(
        target_text,
        max_length=max_target_len,
        padding="max_length",
        truncation=True
    )

    input_encoding["labels"] = target_encoding["input_ids"]
    return input_encoding

# Tokenize
tokenized_train = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names)
tokenized_val = val_data.map(preprocess, batched=True, remove_columns=val_data.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5-cnn-model",
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    eval_steps=2_000,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    learning_rate=3e-4,
    fp16=True,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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

# Save final model
model.save_pretrained("./t5-cnn-final")
tokenizer.save_pretrained("./t5-cnn-final")
