"""
Judgment OCR and Cleaning Pipeline
----------------------------------
This script performs OCR (Optical Character Recognition) on PDF judgment files,
extracts text using Tesseract, cleans it using NLP and regex, splits the cleaned text
into sentences, and stores everything in a structured JSON file.

Major Steps:
1. Read judgment metadata (PDF links) from JSON
2. Extract text from each PDF using OCR
3. Clean and normalize text for NLP
4. Split cleaned text into sentences
5. Save extracted text, cleaned text, and sentence list into a JSON file
"""

# ==== IMPORTS ====
import os
import json
import re
import nltk
import spacy
from pdf2image import convert_from_path
import pytesseract
from nltk.tokenize import sent_tokenize

# ==== INITIAL SETUP ====
# Ensure required NLTK tokenizers are available
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

# Load SpaCy English model (useful for future NLP tasks)
nlp = spacy.load("en_core_web_sm")

# ==== CONFIGURATION ====
INPUT_JSON = "judgments_2025.json"
OUTPUT_JSON = "judgments_with_ocr_sentences_2025.json"
PDF_DIR = "judgments"

# Paths for dependencies (Windows)
POPPLER_PATH = r"C:\poppler\poppler-25.07.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ==== TEXT CLEANING FUNCTION ====
def clean_text(text):
    """
    Extracts only the relevant 'JUDGMENT' or 'ORDER' section from text,
    removes unwanted noise, and keeps essential numerics, dates, and legal references.

    Args:
        text (str): Raw OCR text from PDF

    Returns:
        cleaned_text (str): Preprocessed and cleaned text ready for NLP tasks
    """

    # 1️⃣ Normalize whitespace and newlines
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

    # 2️⃣ Detect the start of 'JUDGMENT' or 'ORDER' section
    pattern = re.compile(r"\b(J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|O\s*R\s*D\s*E\s*R)\b")
    match = pattern.search(text)
    if match:
        text = text[match.start():]
    else:
        text = "SECTION NOT FOUND: " + text[:2000]

    # 3️⃣ Remove repetitive metadata or artifacts
    text = re.sub(r"\bPage\s*\d+(\s*of\s*\d+)?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bJAIL PETITION NO\.\s*\d+\s*OF\s*\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bConst\.?P\.?\s*\d+/?\d*\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[|_•¬]+", " ", text)

    # 4️⃣ Protect important legal entities and numerics before cleaning
    protected = {}
    pattern_keep = re.compile(
        r'\b(?:PLD\s*\d{4}\s*\w+\s*\d+|SCMR\s*\d{4}\s*\d+|Article\s*\d+[A-Za-z]*|Section\s*\d+[A-Za-z]*|[0-3]?\d[\./-][01]?\d[\./-]\d{2,4}|[A-Z]+\s*\d+/\d+)\b'
    )
    i = 0

    def protect(match):
        nonlocal i
        key = f"__PROTECTED_{i}__"
        protected[key] = match.group(0)
        i += 1
        return key

    text = pattern_keep.sub(protect, text)

    # 5️⃣ Remove unwanted symbols (retain letters, numbers, punctuation)
    text = re.sub(r"[^\w\s.,;:!?/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 6️⃣ Restore protected patterns
    for key, value in protected.items():
        text = text.replace(key, value)

    # 7️⃣ Fix OCR hyphenation (e.g., "consti- tution" → "constitution")
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    return text.strip()


# ==== OCR EXTRACTION FUNCTION ====
def extract_text_from_pdf(pdf_path):
    """
    Converts each PDF page into an image and extracts text using Tesseract OCR.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        text (str): Extracted raw OCR text (cleaned page by page)
    """
    try:
        pages = convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)
        text = ""
        for page in pages:
            page_text = pytesseract.image_to_string(page, lang='eng')
            text += page_text + "\n"
        return text.strip()

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def fix_sentence_splits(sentences):
    """
    Merge sentence fragments caused by incorrect sentence splits from OCR or tokenization.
    Handles:
      - Currency splits (e.g., 'Rs.' + '500,000')
      - Registration/Case No. splits (e.g., 'Registration No.' + 'JY-3456')
      - Very short fragments followed by numbers or uppercase tokens
    """

    merged = []
    i = 0

    # regexes for common cases
    starts_with_number_or_id = re.compile(
        r'^[\(\[]?\s*(?:\d+|[A-Z]{1,4}[-/]\d+|[A-Z]{2,5}\d+|JY-|FIR|PLD|SCMR)', re.IGNORECASE
    )
    ends_with_currency = re.compile(
        r'(?:\bRs\.?|\bRupees|\bPKR\b|\bINR\b|Rs/|Rs:|\bRs\b)$', re.IGNORECASE
    )
    ends_with_no = re.compile(r'\bNo\.$', re.IGNORECASE)

    while i < len(sentences):
        s = sentences[i].strip()
        if i + 1 < len(sentences):
            nxt = sentences[i + 1].strip()

            # ---- RULE 1: join if sentence ends with "No." (Registration No., Case No., etc.) ----
            if ends_with_no.search(s) and starts_with_number_or_id.match(nxt):
                s = s + " " + nxt
                i += 2
                merged.append(s.strip())
                continue

            # ---- RULE 2: currency-based join ("Rs." + "500,000") ----
            if ends_with_currency.search(s):
                s = s + " " + nxt
                i += 2
                # keep merging if next starts with number/id (to handle multi-part fragments)
                while i < len(sentences) and starts_with_number_or_id.match(sentences[i].strip()):
                    s = s + " " + sentences[i].strip()
                    i += 1
                merged.append(s.strip())
                continue

            # ---- RULE 3: very short fragments followed by number/ID/currency ----
            if len(s.split()) < 4 and starts_with_number_or_id.match(nxt):
                s = s + " " + nxt
                i += 2
                merged.append(s.strip())
                continue

        # default case — keep as-is
        merged.append(s)
        i += 1

    return merged


# ==== MAIN PIPELINE ====
def main():
    """
    Main driver function.
    1. Reads JSON file with judgments and PDF URLs
    2. Performs OCR extraction for each PDF
    3. Cleans the extracted text
    4. Splits cleaned text into sentences
    5. Stores results in a new JSON file
    """

    # Load input JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for judgment in data["Judgments"]:
        pdf_path = judgment.get("Download")
        if not pdf_path:
            continue

        full_pdf_path = os.path.join(PDF_DIR, os.path.basename(pdf_path))
        print(f"Processing {full_pdf_path} ...")

        # Step 1: Extract text via OCR
        extracted_text = extract_text_from_pdf(full_pdf_path)

        if extracted_text:
            # Step 2: Clean text
            cleaned_text = clean_text(extracted_text)

            # Step 3: Split into sentences (for NLP training)
            sentences = sent_tokenize(cleaned_text)

            # Step 4: Filter out very short or broken sentences initially
            raw_sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

            # Step 5: Fix common wrong splits (join fragments like "Rs." + "500,000/- ...")
            sentences = fix_sentence_splits(raw_sentences)

            # Step 6: Final filtering (keep only meaningful sentences)
            sentences = [s for s in sentences if len(s.split()) > 5]

            # Step 5: Store results in structured format
            judgment["ExtractedText"] = extracted_text
            judgment["CleanedText"] = cleaned_text
            judgment["Sentences"] = sentences

        else:
            judgment["ExtractedText"] = "ERROR: Could not extract text"
            judgment["CleanedText"] = ""
            judgment["Sentences"] = []

    # Write output JSON file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("\n✅ OCR + Cleaning + Sentence Splitting complete.")
    print(f"Output saved to {OUTPUT_JSON}")


# ==== SCRIPT ENTRY POINT ====
if __name__ == "__main__":
    main()
