import json
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

def extract_text_from_pdf_or_image(file_path: Path) -> str:
    """
    Extracts text from a scanned PDF or image using Tesseract OCR.
    """
    extracted_pages = []
    if file_path.suffix.lower() == ".pdf":
        images = convert_from_path(str(file_path))
        for img in images:
            text = pytesseract.image_to_string(img)
            extracted_pages.append(text)
    else:
        from PIL import Image
        img = Image.open(str(file_path))
        text = pytesseract.image_to_string(img)
        extracted_pages.append(text)
    return "\n".join(extracted_pages)

def run_ocr_on_eval_subset(subset_dir="./data/ocr_eval_subset"):
    """
    Processes all raw scans in data/ocr_eval_subset/raw_scans
    and saves normalized text to data/ocr_eval_subset/ocr_extracted.json.
    """
    base_dir = Path(subset_dir)
    scans_dir = base_dir / "raw_scans"
    output_path = base_dir / "ocr_extracted.json"
    
    results = []
    if not scans_dir.exists():
        print(f"Scans directory not found: {scans_dir}")
        return results

    scan_files = list(scans_dir.glob("*.pdf")) + list(scans_dir.glob("*.png")) + list(scans_dir.glob("*.jpg"))
    print(f"Processing {len(scan_files)} files in OCR evaluation subset...")

    for file_path in scan_files:
        doc_id = file_path.stem
        text = extract_text_from_pdf_or_image(file_path)
        results.append({
            "id": doc_id,
            "ocr_text": text
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"OCR text extraction complete. Saved to {output_path}")
    return results

if __name__ == "__main__":
    run_ocr_on_eval_subset()