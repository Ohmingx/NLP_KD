import os
import json
import torch
import evaluate
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.preprocessing.data_loader import load_multilexsum

# --- CHANGED: Use environment variables to allow Kaggle overrides ---
# Defaults to your local paths if not running in Kaggle
MODEL_PATH = os.getenv("MODEL_PATH", "./saved_models/final_teacher_model")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./reports/teacher_eval_v1")

SPLIT_TO_EVAL = "test"   
TARGET_GRANULARITY = "summary/short"
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 256

def extract_text(sources, words_limit=950):
    full_text = " ".join(sources) if isinstance(sources, list) else str(sources)
    words = full_text.split()
    if len(words) > words_limit:
        mid_start = int(len(words) * 0.15)
        return " ".join(words[:250]) + " ... [TEXT OMITTED] ... " + " ".join(words[mid_start:mid_start+700])
    return full_text

def novel_ngram_ratio(source, summary, n=3):
    src_ngrams = set(zip(*[source.split()[i:] for i in range(n)]))
    sum_words = summary.split()
    sum_ngrams = list(zip(*[sum_words[i:] for i in range(n)]))
    if not sum_ngrams:
        return 0.0
    novel = sum(1 for g in sum_ngrams if g not in src_ngrams)
    return novel / len(sum_ngrams)

def repetition_rate(summary, n=3):
    words = summary.split()
    ngrams = list(zip(*[words[i:] for i in range(n)]))
    if not ngrams:
        return 0.0
    return 1 - (len(set(ngrams)) / len(ngrams))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    raw_data = load_multilexsum()
    cases = [
        c for c in raw_data[SPLIT_TO_EVAL]
        if c.get(TARGET_GRANULARITY) and c.get("sources")
    ]
    print(f"Evaluating on {len(cases)} examples from '{SPLIT_TO_EVAL}' split.")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Loading metrics...")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    predictions, references, source_texts = [], [], []

    print("Generating summaries...")
    for i, case in enumerate(cases):
        source_text = extract_text(case["sources"])
        input_text = "summarize civil rights legal document: " + source_text
        inputs = tokenizer(
            input_text, return_tensors="pt",
            max_length=MAX_SOURCE_LENGTH, truncation=True
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LENGTH,
                min_new_tokens=25,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        predictions.append(pred)
        references.append(case[TARGET_GRANULARITY].strip())
        source_texts.append(source_text)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cases)} done")

    print("Computing metrics...")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=[[r] for r in references])
    meteor_scores = meteor.compute(predictions=predictions, references=references)
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")

    novel_ratios = [novel_ngram_ratio(s, p) for s, p in zip(source_texts, predictions)]
    rep_rates = [repetition_rate(p) for p in predictions]
    len_ratios = [len(p.split()) / max(len(r.split()), 1) for p, r in zip(predictions, references)]

    summary_metrics = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "rougeLsum": rouge_scores["rougeLsum"],
        "bleu": bleu_scores["bleu"],
        "meteor": meteor_scores["meteor"],
        "bertscore_precision": sum(bert_scores["precision"]) / len(bert_scores["precision"]),
        "bertscore_recall": sum(bert_scores["recall"]) / len(bert_scores["recall"]),
        "bertscore_f1": sum(bert_scores["f1"]) / len(bert_scores["f1"]),
        "avg_novel_3gram_ratio": sum(novel_ratios) / len(novel_ratios),
        "avg_repetition_rate": sum(rep_rates) / len(rep_rates),
        "avg_length_ratio_pred_to_ref": sum(len_ratios) / len(len_ratios),
        "num_examples": len(cases),
    }

    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    for k, v in summary_metrics.items():
        print(f"{k:30s}: {v}")

    with open(os.path.join(OUTPUT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)

    per_example_df = pd.DataFrame({
        "prediction": predictions,
        "reference": references,
        "novel_3gram_ratio": novel_ratios,
        "repetition_rate": rep_rates,
        "length_ratio": len_ratios,
        "bertscore_f1": bert_scores["f1"],
    })
    per_example_df.to_csv(os.path.join(OUTPUT_DIR, "per_example_results.csv"), index=False)

    print(f"\nSaved detailed results to {OUTPUT_DIR}/")
    print("Worst 5 examples by BERTScore F1 (good candidates to inspect manually):")
    print(per_example_df.sort_values("bertscore_f1").head(5)[["reference", "prediction", "bertscore_f1"]])

if __name__ == "__main__":
    main()