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

def train_baseline():
    print("=== STAGE 5: FLAN-T5 Small Baseline Student ===")
    
    raw_data = load_multilexsum()
    tokenized_dataset, tokenizer = build_smart_tokenized_dataset(
        raw_data, 
        tokenizer_name="google/flan-t5-small",
        target_granularity="summary/short"
    )

    model_name = "google/flan-t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/flan-t5-small-baseline",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,       # Can be slightly higher for the small model
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,       # Adjusting to maintain effective batch size 16
        gradient_checkpointing=True,
        optim="adafactor",
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=False,
        fp16=False,
        logging_dir="./logs/baseline",
        logging_steps=50,
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

    trainer.train()
    
    os.makedirs("./saved_models/final_baseline_student", exist_ok=True)
    trainer.save_model("./saved_models/final_baseline_student")
    tokenizer.save_pretrained("./saved_models/final_baseline_student")
    print("Baseline student adaptation complete. Checkpoint saved.")

if __name__ == "__main__":
    train_baseline()