import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from src.preprocessing.data_loader import load_multilexsum
from src.preprocessing.tokenization import build_smart_tokenized_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_baseline_student():
    print("=== STAGE 5: FLAN-T5 Small Baseline Adaptation (Supervised) ===")
    
    raw_data = load_multilexsum()
    tokenized_dataset, tokenizer = build_smart_tokenized_dataset(
        raw_data, 
        tokenizer_name="google/flan-t5-small",
        target_granularity="summary/short"
    )
    
    model_name = "google/flan-t5-small"
    print(f"Loading student baseline model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    output_dir = "./checkpoints/flan-t5-small-baseline"
    final_save_path = "./saved_models/final_baseline_student"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        optim="adafactor",
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=False,
        fp16=False,                # <-- Set to False to prevent T5 FP16 NaN explosion
        bf16=False,                # <-- Keep False unless running on an Ampere GPU (A100/H100)
        logging_dir="./logs/student_baseline",
        logging_steps=20,
        report_to="none"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    print("Starting baseline student training...")
    trainer.train()
    
    os.makedirs(final_save_path, exist_ok=True)
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Baseline student training complete. Model saved to {final_save_path}")

if __name__ == "__main__":
    train_baseline_student()