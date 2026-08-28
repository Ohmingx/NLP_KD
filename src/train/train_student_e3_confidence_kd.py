import os
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from src.preprocessing.data_loader import load_multilexsum
from src.preprocessing.tokenization import build_smart_tokenized_dataset
from src.train.kd_trainer import DistillationTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TEACHER_PATH = "./saved_models/final_teacher_model"
STUDENT_MODEL_NAME = "google/flan-t5-small"
ALPHA = 0.5
TEMPERATURE = 2.0
CONFIDENCE_METHOD = "entropy"   # or "margin"
WEIGHT_POWER = 1.0              # try 2.0 for a sharper confidence contrast later


def train_student_e3():
    print("=== STAGE 5 (E3): Confidence-Aware Knowledge Distillation ===")

    raw_data = load_multilexsum()
    tokenized_dataset, tokenizer = build_smart_tokenized_dataset(
        raw_data,
        tokenizer_name=STUDENT_MODEL_NAME,
        target_granularity="summary/short"
    )

    print(f"Loading frozen teacher from: {TEACHER_PATH}")
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_PATH)

    print(f"Loading student: {STUDENT_MODEL_NAME}")
    student_model = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL_NAME)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=student_model)

    # Identical hyperparameters to E2 -- only the loss weighting differs
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/flan-t5-small-e3-confidence-kd",
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
        fp16=False,
        bf16=False,
        seed=42,
        data_seed=42,
        logging_dir="./logs/student_e3_confidence_kd",
        logging_steps=20,
        report_to="none"
    )

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        alpha=ALPHA,
        temperature=TEMPERATURE,
        confidence_aware=True,               # <-- E3: per-token weighted KD
        confidence_method=CONFIDENCE_METHOD,
        weight_power=WEIGHT_POWER,
        weight_clip=(0.1, 5.0),
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator
    )

    print("Starting E3 confidence-aware KD training...")
    trainer.train()

    save_path = "./saved_models/final_student_e3_confidence_kd"
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"E3 training complete. Saved to {save_path}")


if __name__ == "__main__":
    train_student_e3()