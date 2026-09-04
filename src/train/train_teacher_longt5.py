import os
import numpy as np
import evaluate
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoConfig,
)

from src.preprocessing.data_loader import load_multilexsum
from src.preprocessing.document_selection import build_typed_tokenized_dataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_NAME = "google/long-t5-tglobal-large"
MAX_SOURCE_LENGTH = 2048
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
            tokenizer.pad_token_id
        )

        decoded_preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True
        )

        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id
        )

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        result = {
            k: round(v, 4)
            for k, v in result.items()
        }

        pred_lens = [
            len(
                tokenizer(
                    p,
                    add_special_tokens=False
                )["input_ids"]
            )
            for p in decoded_preds
        ]

        result["gen_len"] = float(np.mean(pred_lens))

        return result

    return compute_metrics


def train_teacher():

    data = load_multilexsum(
        raw_dir_path="./data/raw",
        parsed_cache_path="./data/raw/multilexsum_parsed.pkl",
    )

    tokenized_dataset, tokenizer = build_typed_tokenized_dataset(
        parsed_data=data,
        tokenizer_name=MODEL_NAME,
        target_granularity="summary/short",
        max_source_length=MAX_SOURCE_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
    )

    # 1. FILTER EMPTY DATA (Prevents loss: 0 on blank summaries)
    tokenized_dataset["train"] = tokenized_dataset["train"].filter(
        lambda x: len(x["input_ids"]) > 2 and len(x["labels"]) > 2
    )
    tokenized_dataset["validation"] = tokenized_dataset["validation"].filter(
        lambda x: len(x["input_ids"]) > 2 and len(x["labels"]) > 2
    )

    # 2. FIX THE CONFIG
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.tie_word_embeddings = True

    # 3. LOAD NATIVE PYTORCH WEIGHTS (Bypasses Flax & Safetensors bugs)
    print("Loading base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, 
        config=config,
        use_safetensors=False 
    )

    # 4. SMOKE TEST
    print("\n--- RUNNING EMBEDDING SMOKE TEST ---")
    test_in = tokenizer("summarize: The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    out = model.generate(**test_in, max_new_tokens=20)
    smoke_test_result = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Output: {smoke_test_result}")
    print("------------------------------------\n")

    # 5. APPLY LORA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8, 
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"] 
    )
    model = get_peft_model(model, lora_config)

    # Required for gradient checkpointing to work alongside LoRA
    model.enable_input_require_grads()
    model.print_trainable_parameters()


    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/long-t5-large-teacher",

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=1e-4,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=16,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        max_grad_norm=1.0,
        warmup_ratio=0.05,

        optim="adamw_torch",
        weight_decay=0.01,

        save_total_limit=2,

        num_train_epochs=2,

        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=2,

        metric_for_best_model="rouge1",
        greater_is_better=True,
        load_best_model_at_end=True,

        fp16=False,
        bf16=False,
        
        logging_dir="./logs/teacher",
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

    final_path = "./saved_models/final_teacher_longt5_large"

    os.makedirs(final_path, exist_ok=True)

    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)


if __name__ == "__main__":
    train_teacher()