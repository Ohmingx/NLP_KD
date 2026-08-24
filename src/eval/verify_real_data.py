import random
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.preprocessing.data_loader import load_multilexsum

def verify_on_real_data():
    print("Loading Dataset...")
    raw_data = load_multilexsum()
    
    # Filter out cases where 'summary/short' is None or empty
    valid_val_cases = [
        item for item in raw_data["validation"]
        if item.get("summary/short") and len(item["summary/short"].strip()) > 0
    ]
    
    if not valid_val_cases:
        print("No valid cases with 'summary/short' found in validation split!")
        return

    # Select a valid random sample
    sample = random.choice(valid_val_cases)
    
    # Handle sources whether stored as string or list of strings
    raw_sources = sample["sources"]
    if isinstance(raw_sources, list):
        source_text = " ".join(raw_sources)
    else:
        source_text = str(raw_sources)

    human_summary = sample["summary/short"]
    
    print(f"Loading Teacher Model from local storage...")
    model_path = "./saved_models/final_teacher_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    # Prompt formatting matching Multi-LexSum summarization
    input_text = f"summarize legal case: {source_text[:3500]}"
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    )
    
    print("\nGenerating model summary...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            min_new_tokens=25,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    model_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("=" * 80)
    print("GROUND TRUTH HUMAN SUMMARY:")
    print("-" * 80)
    print(human_summary.strip())
    print("=" * 80)
    print("TEACHER MODEL SUMMARY:")
    print("-" * 80)
    print(model_summary.strip())
    print("=" * 80)

if __name__ == "__main__":
    verify_on_real_data()