from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def build_smart_tokenized_dataset(
    parsed_data, 
    tokenizer_name="google/flan-t5-small", 
    target_granularity="summary/short", 
    max_source_length=1024, 
    max_target_length=256
):
    print(f"Initializing tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hf_splits = {}
    for split_name, cases in parsed_data.items():
        inputs = []
        targets = []
        for case in cases:
            target_summary = case.get(target_granularity)
            sources = case.get("sources")
            if not target_summary or not sources:
                continue
            full_text = " ".join(sources)
            words = full_text.split()
            if len(words) > 950:
                mid_start = int(len(words) * 0.15)
                extracted_text = " ".join(words[:250]) + " ... [TEXT OMITTED] ... " + " ".join(words[mid_start:mid_start+700])
            else:
                extracted_text = full_text
            prompt = "summarize civil rights legal document: " + extracted_text
            inputs.append(prompt)
            targets.append(target_summary)
        hf_splits[split_name] = Dataset.from_dict({
            "document": inputs,
            "summary": targets
        })
        print(f"  {split_name.capitalize()}: {len(targets)} valid cases processed.")
    dataset_dict = DatasetDict(hf_splits)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["document"],
            max_length=max_source_length,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            text_target=examples["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
        # Mask pad tokens so they are ignored in the loss computation
        labels["input_ids"] = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset splits...")
    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=["document", "summary"]
    )
    print("Tokenization complete.")
    return tokenized_datasets, tokenizer