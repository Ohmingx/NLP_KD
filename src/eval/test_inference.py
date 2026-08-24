import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def test_teacher_model():
    print("Loading the Teacher Model and Tokenizer from local storage...")
    model_path = "./saved_models/final_teacher_model"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    # Sample containing standard case header info typical of Multi-LexSum
    sample_document = """
   Case Background & Parties involvedOn November 8, 2025, Apex Analytics Inc., a Delaware-incorporated machine learning enterprise with principal offices in Boston, Massachusetts, initiated a high-stakes civil lawsuit in the United States District Court for the District of Massachusetts. The named defendant is Dr. Aris Thorne, a former Principal AI Architect who served as the lead designer for Apex’s proprietary predictive forecasting engine, "Aegis-9." Dr. Thorne resigned from the company abruptly on October 12, 2025, giving only a 48-hour notice, and immediately assumed the role of Chief Technology Officer (CTO) at Horizon Data Systems, a direct market competitor based in Austin, Texas.The AllegationsAccording to the filed complaint, Apex Analytics alleges that in the final two weeks of his employment, Dr. Thorne deliberately bypassed internal security protocols to download over 45 gigabytes of restricted intellectual property. Forensic digital audits conducted by Apex’s IT security team revealed that Dr. Thorne connected an unauthorized encrypted solid-state drive (SSD) to his corporate workstation outside of standard working hours. The files allegedly copied include the uncompiled source code for the Aegis-9 core neural network, proprietary data pre-processing pipelines, and a highly confidential directory containing financial pricing models and upcoming product roadmaps spanning through 2028. Furthermore, the plaintiff asserts that Dr. Thorne wiped his corporate-issued laptop using commercial data-deletion software prior to returning it, in direct violation of the company's device management policy.Legal Claims and DemandsThe plaintiff’s legal team has brought forward multiple causes of action against Dr. Thorne, including breach of the Employee Invention Assignment and Non-Disclosure Agreement, violation of the Federal Defend Trade Secrets Act (DTSA), and breach of fiduciary duty. Apex Analytics claims that the potential deployment of Aegis-9's architecture by Horizon Data Systems would cause irreversible market devaluation and erase their three-year technological advantage. Consequently, Apex is petitioning the court for an emergency Temporary Restraining Order (TRO) and a subsequent preliminary injunction to bar Dr. Thorne from performing any technical work related to predictive modeling at Horizon Data Systems. Additionally, the lawsuit seeks punitive financial damages, the immediate return or destruction of all copied assets, and a court-mandated independent forensic audit of Horizon Data Systems' local networks and cloud repositories.
    
    """
    
    # Task prefix matching standard T5 summarization
    input_text = f"summarize: {sample_document.strip()}"
    
    print("\nTokenizing input...")
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    )
    
    print("Generating refined summary...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            min_new_tokens=35,
            num_beams=4,
            length_penalty=1.0,
            repetition_penalty=1.2,    # Smooth penalty instead of a hard 2-gram ban
            no_repeat_ngram_size=3,    # Allows common 2-grams like 'in the' or 'of the'
            early_stopping=True
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("ORIGINAL DOCUMENT:")
    print(sample_document.strip())
    print("-" * 50)
    print("TEACHER MODEL SUMMARY:")
    print(summary)
    print("="*50)

if __name__ == "__main__":
    test_teacher_model()