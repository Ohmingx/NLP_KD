import os

import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from src.preprocessing.data_loader import load_multilexsum
from src.preprocessing.document_selection import build_typed_tokenized_dataset
from src.train.qgh_kd_trainer import QGHKDTrainer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


TEACHER_PATH = "./saved_models/final_teacher_longt5_large"
STUDENT_MODEL_NAME = "google/long-t5-tglobal-base"

MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 256

ALPHA = 0.6
BETA = 0.4
GAMMA = 0.0

TEMPERATURE = 1.5
TOP_K = 20


def train_student_standard_topk_kd():

    raw_data = load_multilexsum()

    tokenized_dataset, tokenizer = build_typed_tokenized_dataset(
        raw_data,
        tokenizer_name=STUDENT_MODEL_NAME,
        target_granularity="summary/short",
        max_source_length=MAX_SOURCE_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
    )

    student = AutoModelForSeq2SeqLM.from_pretrained(
        STUDENT_MODEL_NAME
    )

    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER_PATH
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=student,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/long-t5-base-standard-topk-kd",

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=5e-5,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,

        gradient_accumulation_steps=8,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        max_grad_norm=1.0,
        warmup_ratio=0.05,

        optim="adafactor",
        weight_decay=0.01,

        save_total_limit=2,

        num_train_epochs=4,

        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,

        metric_for_best_model="rouge1",
        greater_is_better=True,

        bf16=torch.cuda.is_bf16_supported(),

        seed=42,
        data_seed=42,

        logging_dir="./logs/student_standard_topk_kd",
        logging_steps=20,

        report_to="none",
    )

    trainer = QGHKDTrainer(
        model=student,
        teacher_model=teacher,

        tokenizer_for_pseudo=tokenizer,

        # No quality gates.
        example_gates=None,

        # No sequence-level KD.
        pseudo_summaries=None,

        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,

        temperature=TEMPERATURE,
        top_k=TOP_K,
        max_target_length=MAX_TARGET_LENGTH,

        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    final_path = (
        "./saved_models/"
        "final_student_standard_topk_kd"
    )

    os.makedirs(
        final_path,
        exist_ok=True,
    )

    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)


if __name__ == "__main__":
    train_student_standard_topk_kd()