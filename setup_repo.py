import os

# 1. Define the directory structure
directories = [
    "data/raw",                  # Multi-LexSum splits
    "data/ocr_eval_subset",      # Scanned documents for the OCR extension
    "src/preprocessing",         # Chunking & tokenisation
    "src/ocr",                   # Image/PDF -> OCR engine -> normalized text
    "src/models",                # Teacher and student model wrappers
    "src/train",                 # baseline.py, standard_kd.py, confidence_kd.py
    "src/eval",                  # rouge.py, efficiency.py, wer_cer.py
    "notebooks",                 # EDA and prototyping only
    "reports",                   # Final outputs and graphs
    "checkpoints",               # Temporary training saves
    "saved_models"               # Final exported models
]

# 2. Create the directories
print("Creating directory structure...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created: {directory}/")

# 3. Create requirements.txt
requirements = """torch
transformers
datasets==2.19.0
accelerate
evaluate
sentencepiece
rouge_score
numpy
pandas
scikit-learn
matplotlib
tqdm
"""
with open("requirements.txt", "w") as f:
    f.write(requirements)
print("Created: requirements.txt")

# 4. Create .gitignore (Crucial to keep GitHub clean)
gitignore = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]

# Data and Models (Do not push to GitHub!)
data/raw/*
data/ocr_eval_subset/*
checkpoints/*
saved_models/*
*.safetensors
*.bin
*.pkl

# Environments
.env
venv/
env/
"""
with open(".gitignore", "w") as f:
    f.write(gitignore)
print("Created: .gitignore")

# 5. Create the train_teacher.py script
train_teacher_code = """import json
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict

def load_and_preprocess_data(file_paths, tokenizer_name="google/flan-t5-large"):
    print("Loading raw data...")
    hf_splits = {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    for split_name, path in file_paths.items():
        with open(path, "r") as f:
            cases = [json.loads(line) for line in f.read().splitlines()]
            
        inputs, targets = [], []
        for case in cases:
            target_summary = case.get("summary/short")
            sources = [case["sources"][d]["doc_text"] for d in case["case_documents"]]
            
            if not target_summary or not sources:
                continue
                
            words = " ".join(sources).split()
            if len(words) > 950:
                mid_start = int(len(words) * 0.15)
                extracted_text = " ".join(words[:250]) + " ... [TEXT OMITTED] ... " + " ".join(words[mid_start:mid_start+700])
            else:
                extracted_text = " ".join(words)
                
            inputs.append("summarize civil rights legal document: " + extracted_text)
            targets.append(target_summary)
            
        hf_splits[split_name] = Dataset.from_dict({"document": inputs, "summary": targets})
        print(f"{split_name} loaded: {len(targets)} cases.")

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["document"], max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(text_target=examples["summary"], max_length=256, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing data...")
    tokenized_datasets = DatasetDict(hf_splits).map(preprocess_function, batched=True, remove_columns=["document", "summary"])
    return tokenized_datasets, tokenizer

def train_teacher():
    file_paths = {
        "train": "../../data/raw/train.json",
        "validation": "../../data/raw/dev.json"
    }
    
    tokenized_dataset, tokenizer = load_and_preprocess_data(file_paths)
    
    print("Loading FLAN-T5 Large (780M Parameters)...")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = ["\\n".join(pred.strip().split()) for pred in decoded_preds]
        decoded_labels = ["\\n".join(label.strip().split()) for label in decoded_labels]
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir="../../checkpoints/flan-t5-large-teacher",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,  
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,             
        predict_with_generate=True,
        fp16=True,                      
        logging_dir="../../logs",
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Beginning Training on RTX 4080...")
    trainer.train()
    trainer.save_model("../../saved_models/final_teacher_model")
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    train_teacher()
"""

with open("src/train/train_teacher.py", "w") as f:
    f.write(train_teacher_code)
print("Created: src/train/train_teacher.py")

print("\\n✅ Project repository successfully generated!")