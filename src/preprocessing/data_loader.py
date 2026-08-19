import json
import pickle
from pathlib import Path
import requests
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/allenai/multi_lexsum/resolve/main/releases/v20230518"

def download_with_progress(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"Already downloaded: {dest_path.name}")
        return
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))

def load_multilexsum(raw_dir_path="./data/raw", parsed_cache_path="./data/raw/multilexsum_parsed.pkl"):
    raw_dir = Path(raw_dir_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = Path(parsed_cache_path)

    if parsed_path.exists():
        print(f"Loading cached parsed data from {parsed_path}")
        with open(parsed_path, "rb") as f:
            return pickle.load(f)

    files = {"train": "train.json", "dev": "dev.json", "test": "test.json", "sources": "sources.json"}
    for fname in files.values():
        if not (raw_dir / fname).exists():
            print(f"Downloading {fname}...")
            download_with_progress(f"{BASE_URL}/{fname}", raw_dir / fname)

    print("Loading sources.json into memory...")
    with open(raw_dir / "sources.json", "r") as f:
        sources = json.load(f)

    def load_split(path):
        with open(path, "r") as f:
            cases = [json.loads(line) for line in f.read().splitlines()]
        return [{
            "id": c["case_id"],
            "sources": [sources[d]["doc_text"] for d in c["case_documents"] if d in sources],
            "sources_metadata": [
                {
                    "is_ocr": sources[d].get("is_ocr", False),
                    "doc_type": sources[d].get("doc_type", "unknown")
                } for d in c["case_documents"] if d in sources
            ],
            "summary/long": c.get("summary/long"),
            "summary/short": c.get("summary/short"),
            "summary/tiny": c.get("summary/tiny"),
        } for c in cases]

    data = {
        "train": load_split(raw_dir / "train.json"),
        "validation": load_split(raw_dir / "dev.json"),
        "test": load_split(raw_dir / "test.json"),
    }

    with open(parsed_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Parsed dataset successfully cached at {parsed_path}")

    return data