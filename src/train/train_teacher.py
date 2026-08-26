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
def train_teacher():
    print("=== STAGE 4: FLAN-T5 Large Teacher Adaptation (Label Mask + Gradient Fix) ===")
    
    raw_data = load_multilexsum()
    tokenized_dataset, tokenizer = build_smart_tokenized_dataset(
        raw_data, 
        tokenizer_name="google/flan-t5-large",
        target_granularity="summary/short"
    )
    model_name = "google/flan-t5-large"
    print(f"Loading teacher model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/flan-t5-large-teacher",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        optim="adafactor",
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=False,
        fp16=False,
        logging_dir="./logs/teacher",
        logging_steps=10,
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
    print("Starting teacher fine-tuning...")
    trainer.train()
    
    os.makedirs("./saved_models/final_teacher_model", exist_ok=True)
    trainer.save_model("./saved_models/final_teacher_model")
    tokenizer.save_pretrained("./saved_models/final_teacher_model")
    print("Teacher adaptation complete. Checkpoint saved.")
if __name__ == "__main__":
    train_teacher()