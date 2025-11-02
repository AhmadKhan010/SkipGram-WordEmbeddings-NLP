# 🧠 SkipGram-WordEmbeddings-NLP

A comprehensive end-to-end NLP pipeline for legal judgment summarization using custom-built Skip-Gram word embeddings and advanced abstractive summarization techniques.

## 📋 Project Overview

This repository implements a complete natural language processing pipeline that:

1. **Extracts text from PDF court judgments** using OCR
2. **Preprocesses and prepares data** for training (tokenization, cleaning, vocabulary building)
3. **Trains Skip-Gram word embeddings from scratch** using only NumPy (no pretrained models)
4. **Generates extractive summaries** using cosine similarity between sentence embeddings
5. **Creates abstractive summaries** using advanced NLP techniques:
   - TF-IDF keyword extraction
   - Part-of-Speech (POS) tagging
   - Named Entity Recognition (NER)
   - Subject-verb-object relationship extraction
   - Redundancy merging and text generation
6. **Evaluates summary quality** using ROUGE and BLEU metrics

### 🎯 Key Features

- **Pure NumPy Implementation**: Skip-Gram neural network built from scratch with mini-batch gradient descent
- **No Pretrained Models**: Complete training pipeline without Hugging Face, Gensim, or Word2Vec
- **Advanced Abstractive Summarization**: Uses spaCy for linguistic analysis and template-based generation
- **Comprehensive Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L, and BLEU score calculations
- **Multiple Execution Modes**: Train, summarize, or both in a single command
- **Progress Tracking**: Real-time training progress with emoji indicators and detailed logging

---

## 🛠️ Requirements

### Python Version

- Python 3.8 or higher

### Dependencies

Install all required packages:

```bash
pip install numpy pandas nltk spacy pdf2image pytesseract pillow
```

Download spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

### Allowed Libraries

As per project constraints:

- ✅ `numpy`, `pandas`, `nltk`, `spacy`
- ✅ `pdf2image`, `pytesseract`, `Pillow`
- ✅ Standard Python libraries (json, pickle, csv, math, random, os)
- ❌ No pretrained embeddings (Gensim, Word2Vec, Sentence-Transformers, Hugging Face)

---

## 🚀 Quick Start Guide

### Step 1: OCR + Extract Text from PDFs

Extract text from PDF court judgments using OCR:

```bash
python "Data Preprocessing/Extracting_data_from_pdfs.py"
```

**Output**: Individual JSON files containing extracted text from each PDF.

---

### Step 2: Merge Multiple JSON Files (Optional)

Combine multiple JSON outputs into a single JSONL file:

```bash
python "Data Preprocessing/merge_jsonl_script.py"
```

**Output**: Merged JSONL file ready for preprocessing.

---

### Step 3: Preprocess Data for Skip-Gram Training

Tokenize text, build vocabulary, and generate skip-gram training pairs:

```bash
python preprocess_and_save.py
```

**Input**: JSONL file with sentences
**Output**: `preprocessed_data.json` containing:

- Vocabulary and word-to-index mappings
- Skip-gram training pairs (target, context)
- Dataset statistics

---

### Step 4: Train Skip-Gram Model

Train the Skip-Gram model using mini-batch gradient descent:

```bash
python train_skipgram_with_minibatch.py train preprocessed_data.json
```

**Hyperparameters** (customizable in code):

- Embedding dimension: 50
- Window size: 2
- Learning rate: 0.3
- Batch size: 16
- Epochs: 5
- Minimum word frequency: 5

**Output**:

- Trained model: `skipgram_model_lr0.3_bs16.pkl`
- Training log: `training_log.csv`

---

### Step 5: Generate Summaries

#### Option A: Summarize Only (Using Pretrained Model)

```bash
python train_skipgram_with_minibatch.py summarize skipgram_model_lr0.3_bs16.pkl test.jsonl 5
```

**Arguments**:

- `skipgram_model_lr0.3_bs16.pkl`: Path to trained model
- `test.jsonl`: Input file with sentences to summarize
- `5`: Number of sentences for extractive summary (K value)

#### Option B: Train + Summarize in One Command

```bash
python train_skipgram_with_minibatch.py both preprocessed_data.json test.jsonl 5
```

**Output**:

- Extractive summary (top K sentences)
- Abstractive summary (rephrased, concise)
- Evaluation metrics (ROUGE, BLEU)

---

## 📊 Pipeline Architecture

### **STEP 1: Data Preprocessing**

```
PDFs → OCR Extraction → Sentence Splitting → Tokenization → Vocabulary Building → Skip-Gram Pairs
```

### **STEP 2: Skip-Gram Training**

```
Training Pairs → Mini-Batch Gradient Descent → Word Embeddings (W1, W2) → Saved Model
```

### **STEP 3: Extractive Summarization**

```
Sentences → Sentence Embeddings (avg of word embeddings) → Cosine Similarity to Document → Top-K Selection
```

### **STEP 4: Abstractive Summarization**

**Sub-steps**:

1. **TF-IDF Keyword Extraction**

   - Calculate term frequency × inverse document frequency
   - Identify top 15 most important keywords

2. **POS Tagging + NER**

   - Extract nouns (subjects, objects)
   - Extract verbs (actions)
   - Extract adjectives/adverbs (descriptors)
   - Identify named entities (persons, organizations, locations)
   - Extract subject-verb-object relationships

3. **Merge Redundant Information**

   - Deduplicate by lemmatization
   - Merge shared subjects/objects
   - Remove repeated phrases

4. **Generate Abstract Text**
   - Build sentences using subject-verb-object templates
   - Add context using connector words
   - Post-process for grammar and fluency

### **STEP 5: Evaluation**

```
Generated Summary + Reference → ROUGE (1, 2, L) + BLEU → Precision, Recall, F1
```

---

## 📁 Project Structure

```
SkipGram-WordEmbeddings-NLP/
├── Data Preprocessing/
│   ├── Extracting_data_from_pdfs.py      # OCR extraction
│   └── merge_jsonl_script.py             # Merge JSON files
├── Model implementation/
│   ├── preprocess_and_save.py            # Data preprocessing
│   ├── train_skipgram_with_minibatch.py  # Main training & summarization
│   ├──preprocessed_data.json                 # Preprocessed training data
│   ├──test.jsonl                             # Test sentences
│   ├──training_log.csv                       # Training progress log
│   ├──skipgram_model_lr0.3_bs16.pkl         # Trained model
└── README.md                              # This file
```

---

## 💡 Usage Examples

### Example 1: Full Pipeline (Train + Summarize)

```bash
# Step 1: Preprocess data
python preprocess_and_save.py

# Step 2: Train and summarize
python train_skipgram_with_minibatch.py both preprocessed_data.json test.jsonl 3
```

### Example 2: Use Existing Model

```python
from train_skipgram_with_minibatch import summarize_with_model

# Summarize a JSONL file using trained model
summarize_with_model('test.jsonl', 'skipgram_model_lr0.3_bs16.pkl', top_k=5)
```

### Example 3: Training Only

```bash
python train_skipgram_with_minibatch.py train preprocessed_data.json
```

---

## 📈 Understanding the Output

### Extractive Summary

```
📄 Extractive Summary (Top 5 sentences):
   1. Both Committees discounted the order of the Court...
   2. wherein the question of a jurisdictional bar...
   ...
```

### Abstractive Summary

```
🎯 Abstractive Summary:
Committee withdrew cases, highlighting Court. Bench ordered matter, concerning order...
```

### Evaluation Metrics

```
📊 Evaluation Metrics:
   ROUGE-1: Precision=0.45, Recall=0.52, F1=0.48
   ROUGE-2: Precision=0.23, Recall=0.28, F1=0.25
   ROUGE-L: Precision=0.41, Recall=0.48, F1=0.44
   BLEU: 0.31
```

**Interpretation**:

- **ROUGE-1**: Measures unigram (word) overlap
- **ROUGE-2**: Measures bigram (phrase) overlap
- **ROUGE-L**: Measures longest common subsequence
- **BLEU**: Overall n-gram precision with brevity penalty
- Higher scores = better summary quality (closer to reference)

---

## 🔧 Customization

### Modify Hyperparameters

Edit `train_skipgram_with_minibatch.py`:

```python
# Training hyperparameters
learning_rate = 0.3
batch_size = 16
epochs = 5
embedding_dim = 50
window_size = 2
```

### Change Summary Length

```bash
# Get top 10 sentences instead of 5
python train_skipgram_with_minibatch.py summarize model.pkl test.jsonl 10
```

### Adjust Abstractive Summary Settings

```python
# In abstractive_summary function:
top_keywords = calculate_tfidf(extractive_sentences, top_n=20)  # Increase keywords
```

---

## 🎓 Assignment Requirements

This project implements all 5 steps of the NLP assignment:

- ✅ **Step 1**: Data Preprocessing (OCR, cleaning, tokenization)
- ✅ **Step 2**: Skip-Gram Training (NumPy implementation, mini-batch gradient descent)
- ✅ **Step 3**: Extractive Summarization (cosine similarity ranking)
- ✅ **Step 4**: Abstractive Summarization (TF-IDF + POS + NER + text generation)
- ✅ **Step 5**: Evaluation Metrics (ROUGE-1/2/L + BLEU scores)

---

## 🐛 Troubleshooting

### Issue: `KeyError: 'vocab_size'`

**Solution**: Regenerate `preprocessed_data.json` using `preprocess_and_save.py`

### Issue: spaCy model not found

**Solution**:

```bash
python -m spacy download en_core_web_sm
```

### Issue: Out of memory during training

**Solution**: Reduce batch size or embedding dimension in the code

### Issue: Summary is too short/generic

**Solution**:

- Increase `top_k` value (more extractive sentences)
- Increase `top_n` in `calculate_tfidf()` (more keywords)

---

## 📝 Notes

- **Training Time**: ~5-10 minutes depending on dataset size and hardware
- **Model Size**: ~2-5 MB for typical vocabulary
- **Best Practices**:
  - Use at least 100+ documents for meaningful embeddings
  - Ensure test data uses words from training vocabulary
  - Higher `top_k` values provide more context for abstractive summary

---

## 📄 License

This project is created for academic purposes as part of an NLP assignment.

---

## 👨‍💻 Author

Created as part of NLP Assignment 2, Semester 7


---

