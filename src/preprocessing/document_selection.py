import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


DOC_TYPE_PRIORITY = {
    "Order/Opinion": (1.00, "tail"),
    "Complaint": (0.85, "head"),
    "Settlement Agreement": (0.80, "tail"),
    "Findings Letter/Report": (0.65, "tail"),
    "Pleading / Motion / Brief": (0.55, "head"),
    "Declaration/Affidavit": (0.45, "head"),
    "Magistrate Report/Recommendation": (0.45, "tail"),
    "Monitor/Expert/Receiver Report": (0.35, "tail"),
    "Press Release": (0.30, "head"),
    "Discovery Material/FOIA Release": (0.25, "head"),
    "Statute/Ordinance/Regulation": (0.25, "head"),
    "Internal memorandum": (0.20, "head"),
    "Docket": (0.10, "head"),
    "Notice Letter": (0.15, "head"),
    "Correspondence": (0.10, "head"),
    "Other": (0.15, "head"),
    "unknown": (0.15, "head"),
}

DEFAULT_PRIORITY = (0.15, "head")


OUTCOME_QUERY = (
    "ordered adjudged decreed holding ruling judgment relief damages "
    "injunction settlement statute violation plaintiff defendant class "
    "certified consent decree remedy court finds concludes"
)


_OCR_NOISE_PATTERNS = [
    (re.compile(r"[\.\-_]{4,}"), " "),
    (
        re.compile(r"\b[A-Za-z]\s(?=[A-Za-z]\s[A-Za-z])"),
        "",
    ),
    (re.compile(r"\s{2,}"), " "),
    (
        re.compile(
            r"Case\s+\d[\d:\-a-zA-Z]*\s+Document\s+\d+.*?"
            r"Page\s+\d+\s+of\s+\d+",
            re.IGNORECASE,
        ),
        " ",
    ),
]


def clean_ocr_text(text: str) -> str:
    """Remove common OCR and docket-stamp artifacts."""
    for pattern, replacement in _OCR_NOISE_PATTERNS:
        text = pattern.sub(replacement, text)

    return text.strip()


def _select_words(
    words: List[str],
    n_words: int,
    side: str,
) -> List[str]:

    if len(words) <= n_words:
        return words

    if side == "head":
        return words[:n_words]

    if side == "tail":
        return words[-n_words:]

    raise ValueError(f"Unknown side: {side}")


def _tfidf_rank_sentences(
    sentences: List[str],
    query: str,
    keep_ratio: float = 0.85,
) -> List[str]:

    if len(sentences) <= 3:
        return sentences

    corpus = sentences + [query]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
        )

        tfidf = vectorizer.fit_transform(corpus)

    except ValueError:
        return sentences

    query_vec = tfidf[-1]
    doc_vecs = tfidf[:-1]

    scores = np.asarray(
        (doc_vecs @ query_vec.T).todense()
    ).ravel()

    n_keep = max(
        3,
        int(len(sentences) * keep_ratio),
    )

    top_idx = sorted(
        np.argsort(-scores)[:n_keep]
    )

    return [
        sentences[i]
        for i in top_idx
    ]


def build_case_text(
    sources: List[str],
    metadata: List[dict],
    token_budget_words: int = 3800,
    apply_tfidf_rerank: bool = True,
) -> str:

    docs = []

    for i, text in enumerate(sources):

        text = clean_ocr_text(text or "")

        if not text:
            continue

        doc_type = "unknown"

        if i < len(metadata) and metadata[i]:
            doc_type = (
                metadata[i].get("doc_type")
                or "unknown"
            )

        priority, side = DOC_TYPE_PRIORITY.get(
            doc_type,
            DEFAULT_PRIORITY,
        )

        docs.append(
            {
                "text": text,
                "doc_type": doc_type,
                "priority": priority,
                "side": side,
            }
        )

    if not docs:
        return ""

    # Highest-priority documents first.
    docs.sort(
        key=lambda d: -d["priority"]
    )

    remaining_words = token_budget_words
    selected_chunks = []

    for d in docs:

        if remaining_words <= 0:
            break

        alloc = min(
            remaining_words,
            max(
                150,
                int(
                    remaining_words
                    * d["priority"]
                    * 0.6
                ),
            ),
        )

        words = d["text"].split()

        chunk_words = _select_words(
            words,
            alloc,
            d["side"],
        )

        chunk_text = " ".join(chunk_words)

        selected_chunks.append(
            (
                d["doc_type"],
                chunk_text,
            )
        )

        remaining_words -= len(chunk_words)

    if apply_tfidf_rerank:

        all_sentences = []

        for _, chunk_text in selected_chunks:

            all_sentences.extend(
                re.split(
                    r"(?<=[.!?])\s+",
                    chunk_text,
                )
            )

        all_sentences = [
            sentence
            for sentence in all_sentences
            if len(sentence.split()) >= 3
        ]

        ranked = _tfidf_rank_sentences(
            all_sentences,
            OUTCOME_QUERY,
        )

        return " ".join(ranked)

    return " [SECTION] ".join(
        chunk
        for _, chunk in selected_chunks
    )


def build_typed_tokenized_dataset(
    parsed_data: Dict[str, list],
    tokenizer_name: str = "google/long-t5-tglobal-base",
    target_granularity: str = "summary/short",
    max_source_length: int = 4096,
    max_target_length: int = 256,
    token_budget_words: Optional[int] = None,
) -> Tuple[DatasetDict, AutoTokenizer]:

    print(
        f"Initializing tokenizer: {tokenizer_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name
    )

    if token_budget_words is None:
        token_budget_words = int(
            max_source_length / 1.3
        )

    hf_splits = {}

    for split_name, cases in parsed_data.items():

        inputs = []
        targets = []

        for case in cases:

            target_summary = case.get(
                target_granularity
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
                token_budget_words=token_budget_words,
            )

            if not extracted_text:
                continue

            prompt = (
                "summarize civil rights legal document: "
                + extracted_text
            )

            inputs.append(prompt)
            targets.append(target_summary)

        hf_splits[split_name] = Dataset.from_dict(
            {
                "document": inputs,
                "summary": targets,
            }
        )

        print(
            f"  {split_name.capitalize()}: "
            f"{len(targets)} valid cases processed."
        )

    dataset_dict = DatasetDict(hf_splits)

    def preprocess_function(examples):

        model_inputs = tokenizer(
            examples["document"],
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            text_target=examples["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )

        # Ignore padding tokens during loss calculation.
        labels["input_ids"] = [
            [
                token
                if token != tokenizer.pad_token_id
                else -100
                for token in label
            ]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels[
            "input_ids"
        ]

        return model_inputs

    print("Tokenizing dataset splits...")

    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=[
            "document",
            "summary",
        ],
    )

    print("Tokenization complete.")

    return tokenized_datasets, tokenizer