import os
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from src.preprocessing.data_loader import load_multilexsum
from src.preprocessing.tokenization import build_smart_tokenized_dataset
from src.eval.rouge import build_compute_metrics_fn

def train_baseline_student():
    print("=== STAGE 5: FLAN-T5 Small Supervised Baseline Student (E1) ===")
    
    # 1. Load Data
    raw_data = load_multilexsum()
    tokenized_dataset, tokenizer = build_smart_tokenized_dataset(
        raw_data, 
        tokenizer_name="google/flan-t5-small",
        target_granularity="summary/short"
    )

    # 2. Load Model
    model_name = "google/flan-t5-small"
    print(f"Loading student model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = build_compute_metrics_fn(tokenizer)

    # 3. Configure Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/flan-t5-small-baseline",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        logging_dir="./logs/baseline",
        logging_steps=50,
        report_to="none"
    )

    # 4. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 5. Train and Save Final Checkpoint
    print("Starting baseline training...")
    trainer.train()
    
    os.makedirs("./saved_models/final_baseline_student", exist_ok=True)
    trainer.save_model("./saved_models/final_baseline_student")
    tokenizer.save_pretrained("./saved_models/final_baseline_student")
    print("Baseline student training complete. Saved to ./saved_models/final_baseline_student")

if __name__ == "__main__":
    train_baseline_student()