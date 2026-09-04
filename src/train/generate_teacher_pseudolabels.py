import json
import os

import evaluate
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.preprocessing.data_loader import load_multilexsum
from src.preprocessing.document_selection import build_case_text


TEACHER_PATH = "./saved_models/final_teacher_longt5_large"
OUTPUT_PATH = "./data/processed/teacher_pseudolabels_train.jsonl"

MAX_SOURCE_LENGTH = 8192
MAX_TARGET_LENGTH = 256


rouge = evaluate.load("rouge")


def main():

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    raw_data = load_multilexsum()
    train_cases = raw_data["train"]

    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_PATH
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER_PATH
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    records = []

    with torch.no_grad():

        for idx, case in enumerate(
            tqdm(
                train_cases,
                desc="Generating teacher pseudo-labels",
            )
        ):

            target_summary = case.get(
                "summary/short"
            )

            sources = case.get("sources")

            if not target_summary or not sources:
                continue

            metadata = (
                case.get("sources_metadata")
                or []
            )

            extracted_text = build_case_text(
                sources,
                metadata,
            )

            if not extracted_text:
                continue

            prompt = (
                "summarize civil rights legal document: "
                + extracted_text
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LENGTH,
                min_new_tokens=25,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            pseudo_summary = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            )

            quality = rouge.compute(
                predictions=[pseudo_summary],
                references=[target_summary.strip()],
            )["rougeL"]

            records.append(
                {
                    "example_idx": idx,
                    "pseudo_summary": pseudo_summary,
                    "quality_rougeL": quality,
                }
            )

    if not records:
        raise RuntimeError(
            "No teacher pseudo-label records were generated."
        )

    # Bottom 40% of teacher outputs are gated off.
    scores = sorted(
        r["quality_rougeL"]
        for r in records
    )

    threshold = scores[
        int(len(scores) * 0.40)
    ]

    print(
        f"Quality-gate ROUGE-L threshold "
        f"(40th percentile): {threshold:.4f}"
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            record["gate"] = (
                1.0
                if record["quality_rougeL"] >= threshold
                else 0.0
            )

            f.write(
                json.dumps(record)
                + "\n"
            )

    gated_on = sum(
        r["gate"] == 1.0
        for r in records
    )

    gated_off = sum(
        r["gate"] == 0.0
        for r in records
    )

    print(
        f"Wrote {len(records)} pseudo-label records "
        f"to {OUTPUT_PATH}"
    )

    print(f"Gated ON: {gated_on}")
    print(f"Gated OFF: {gated_off}")


if __name__ == "__main__":
    main()