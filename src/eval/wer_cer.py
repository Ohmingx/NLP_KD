import jiwer

def compute_ocr_error_rates(reference_texts, ocr_texts):
    """
    Computes Word Error Rate (WER) and Character Error Rate (CER)
    between ground truth clean text and OCR-extracted text.
    """
    wer = jiwer.wer(reference_texts, ocr_texts)
    cer = jiwer.cer(reference_texts, ocr_texts)
    return {"WER": round(wer * 100, 2), "CER": round(cer * 100, 2)}