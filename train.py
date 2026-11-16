import torch
from datasets import load_dataset, Features, Value  # <-- Imports 'Features' and 'Value'
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
import evaluate  # Use 'evaluate' library
import os

# --- Configuration ---
DATASET_NAME = "google/code_x_glue_cc_defect_detection"
MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "./codebert-defect-detector"
MAX_LENGTH = 512

# --- 1. Load Dataset ---
print(f"Loading dataset: {DATASET_NAME}")
try:
    dataset = load_dataset(DATASET_NAME)
    print("Dataset loaded successfully.")
    print(dataset)
except Exception as e:
    print(f"Failed to load dataset. Make sure you have internet access. Error: {e}")
    exit()

# --- 2. Load Model and Tokenizer ---
print(f"Loading model and tokenizer: {MODEL_NAME}")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
id2label = {0: "LABEL_0", 1: "LABEL_1"}
label2id = {"LABEL_0": 0, "LABEL_1": 1}

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id,
    problem_type="single_label_classification"
)

# --- 3. Preprocessing ---
print("Tokenizing datasets...")

def tokenize_function(examples):
    return tokenizer(examples["func"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rename 'target' to 'labels'
try:
    tokenized_datasets = tokenized_datasets.rename_column("target", "labels")
    print("Renamed 'target' column to 'labels'.")
except ValueError:
    print("Column 'target' already renamed or does not exist.")


# --- NEW FIX HERE ---
# The .map() function was not enough.
# We must CAST the column's schema from bool to int64 (which is torch.LongTensor)
# This is the correct and final fix for the "expected scalar type Long but found Bool" error.

try:
    # Get the current features
    current_features = tokenized_datasets['train'].features
    
    # Create new features with 'labels' explicitly set to int64 (Long)
    new_features = current_features.copy()
    new_features['labels'] = Value('int64') # 'int64' is what torch.LongTensor expects
    
    # Apply the new features to all splits
    print("Casting 'labels' column's data type to int64 (torch.LongTensor)...")
    tokenized_datasets = tokenized_datasets.cast(new_features)
    
    print("Verifying new features:")
    print(tokenized_datasets['train'].features)

except Exception as e:
    print(f"--- ERROR while casting 'labels' column ---")
    print("This is a critical step. The script will likely fail.")
    print(e)
# --- END OF NEW FIX ---
    

# Remove columns that the model doesn't need
tokenized_datasets = tokenized_datasets.remove_columns(["id", "func", "project", "commit_id"])
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 4. Evaluation Metric ---
print("Loading evaluation metric: f1")
try:
    metric = evaluate.load("f1")
except Exception as e:
    print(f"Failed to load 'f1' metric. Training will proceed without metric calculation. Error: {e}")
    metric = None


def compute_metrics(eval_pred):
    if metric is None:
        return {}
        
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # For defect detection, LABEL_1 is the positive/buggy class.
    # Use binary F1 so the score reflects the buggy class specifically.
    return metric.compute(
        predictions=predictions,
        references=labels,
        average="binary",
        zero_division=0,
    )

# --- 5. Training ---
print("Setting up training arguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Start Training ---
print("\n--- Starting Model Training ---")
print("This will take a while, especially the first time...")
try:
    trainer.train()
    print("\n--- Training Complete ---")

    # --- 6. Save Model ---
    print(f"Saving model and tokenizer to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully!")
    print(f"Your model is ready in the '{OUTPUT_DIR}' folder.")

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED DURING TRAINING ---")
    print(e)
    print("\nTraining was not completed. The model was not saved.")
    print("If this is an 'OutOfMemoryError', try reducing 'per_device_train_batch_size' further (e.g., to 4 or 2).")

# --- 7. Evaluation (Optional) ---
print("\n--- Evaluating on Test Set ---")
try:
    test_results = trainer.evaluate(test_dataset)
    print("Test set results:")
    print(test_results)
except Exception as e:
    print(f"Could not evaluate on test set. Error: {e}")

print("\n--- Script Finished ---")
