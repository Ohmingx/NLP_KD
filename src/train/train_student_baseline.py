import os
import numpy as np
import evaluate
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from src.preprocessing.data_loader import load_multilexsum
from src.preprocessing.document_selection import build_typed_tokenized_dataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_NAME = "google/long-t5-tglobal-base"

MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 256

rouge = evaluate.load("rouge")


def build_compute_metrics(tokenizer):

    def compute_metrics(eval_preds):

        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(
            preds != -100,
            preds,
            tokenizer.pad_token_id,
        )

        decoded_preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
        )

        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id,
        )

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        return {
            k: round(v, 4)
            for k, v in result.items()
        }

    return compute_metrics


def train_baseline_student():

    raw_data = load_multilexsum()

    tokenized_dataset, tokenizer = build_typed_tokenized_dataset(
        raw_data,
        tokenizer_name=MODEL_NAME,
        target_granularity="summary/short",
        max_source_length=MAX_SOURCE_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/long-t5-base-baseline",

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

        num_train_epochs=3,

        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,

        metric_for_best_model="rouge1",
        greater_is_better=True,
        load_best_model_at_end=True,

        bf16=torch.cuda.is_bf16_supported(),

        seed=42,
        data_seed=42,

        logging_dir="./logs/student_baseline",
        logging_steps=20,

        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    final_path = (
        "./saved_models/"
        "final_baseline_student_longt5_base"
    )

    os.makedirs(
        final_path,
        exist_ok=True,
    )

    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)


if __name__ == "__main__":
    train_baseline_student()