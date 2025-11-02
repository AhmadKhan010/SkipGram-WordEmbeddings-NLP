#!/usr/bin/env python
# Skip-Gram word embeddings + Extractive/Abstractive Summarization

# Standard imports
import json
import numpy as np
import time
import pickle
import csv
import os
import pandas as pd
from collections import Counter
import math
import random
import sys

# NLP imports
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import spacy
from preprocess_and_save import load_data

# Load Spacy model for POS tagging and NER
nlp = spacy.load("en_core_web_sm")

# =====================================================
# SKIP-GRAM MODEL
# =====================================================
class SkipGramModel:
    """Predicts context words from target word to learn embeddings"""

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.10  # Input -> Hidden (embeddings)
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.10  # Hidden -> Output
        self.last_loss = None

    def softmax(self, x):
        """Convert scores to probabilities"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=0)

    def forward(self, target_idx):
        """Get embedding and predict context word probabilities"""
        h = self.W1[target_idx]      # Get word embedding
        u = np.dot(h, self.W2)       # Project to output
        y_pred = self.softmax(u)     # Get probabilities
        return y_pred, h

    def backward(self, batch, learning_rate=0.01):
        """Compute gradients and update weights for mini-batch"""
        grad_W1 = np.zeros_like(self.W1)
        grad_W2 = np.zeros_like(self.W2)
        batch_loss = 0.0

        for target, context in batch:
            y_pred, h = self.forward(target)  # Forward pass
            
            y_true = np.zeros(self.vocab_size)  # One-hot encode true context
            y_true[context] = 1.0
            
            error = y_pred - y_true  # Compute error

            grad_W2 += np.outer(h, error)  # Accumulate gradients
            grad_W1[target] += np.dot(self.W2, error)

            batch_loss += -np.log(y_pred[context] + 1e-15)  # Cross-entropy loss

        # Update weights
        batch_size = len(batch)
        if batch_size > 0:
            self.W1 -= (learning_rate / batch_size) * grad_W1
            self.W2 -= (learning_rate / batch_size) * grad_W2

        return batch_loss

    def train(self, samples, epochs=5, learning_rate=0.01, batch_size=512):
        """Train model using mini-batch gradient descent"""
        num_samples = len(samples)
        
        for epoch in range(epochs):
            np.random.shuffle(samples)  # Shuffle for better generalization
            total_loss = 0.0
            print(f"\n🧠 Starting Epoch {epoch + 1}/{epochs}... Total samples: {num_samples}")

            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch = samples[batch_start:batch_end]
                
                batch_loss = self.backward(batch, learning_rate)
                total_loss += batch_loss

                # Show progress
                progress = (batch_end / num_samples) * 100
                if batch_start == 0 or int(progress) % 5 == 0:
                    print(f"   🔹 Progress: {progress:.1f}% complete", end='\r', flush=True)

            avg_loss = total_loss / num_samples if num_samples > 0 else 0
            self.last_loss = avg_loss
            print(f"\n✅ Epoch {epoch + 1}/{epochs} completed. Avg Loss: {avg_loss:.6f}")

    def save(self, filepath, word2idx):
        """Save trained model to file"""
        model_data = {
            'W1': self.W1,
            'W2': self.W2,
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
            'word2idx': word2idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = SkipGramModel(data['vocab_size'], data['embedding_dim'])
        model.W1 = data['W1']
        model.W2 = data['W2']
        
        return model, data.get('word2idx', {})


# Helper functions for training and summarization

def log_results(log_file, params, final_loss, train_time):
    """Log training results to CSV"""
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lr', 'batch_size', 'epochs', 'loss', 'train_time', 'model_path'])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'lr': params['lr'],
            'batch_size': params['batch_size'],
            'epochs': params['epochs'],
            'loss': round(final_loss, 6),
            'train_time': round(train_time, 2),
            'model_path': params['model_path']
        })


def get_sentence_embedding(sentence, word2idx, model):
    """Compute sentence embedding by averaging word embeddings"""
    tokens = [word2idx.get(w.lower(), word2idx.get('<UNK>', 0)) for w in word_tokenize(sentence)]
    embeddings = [model.W1[idx] for idx in tokens if 0 <= idx < model.W1.shape[0]]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.embedding_dim)


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    if norm(v1) == 0 or norm(v2) == 0:  # Handle zero vectors
        return 0
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def extractive_summary(sentences, word2idx, model, k=3):
    """STEP 3: Select top-K sentences most similar to document"""
    sentence_embeddings = [get_sentence_embedding(s, word2idx, model) for s in sentences]
    
    if len(sentence_embeddings) == 0:
        return []
    
    doc_embedding = np.mean(sentence_embeddings, axis=0)  # Overall document vector
    similarities = [cosine_similarity(se, doc_embedding) for se in sentence_embeddings]  # Compare each sentence
    top_k = np.argsort(similarities)[-k:][::-1]  # Get top K indices
    
    # Return the actual sentences
    return [sentences[i] for i in top_k]


# Abstractive summarization functions (STEP 4)

def calculate_tfidf(sentences, top_n=10):
    """Calculate TF-IDF scores to identify important keywords"""
    tokenized_sentences = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        tokenized_sentences.append(tokens)

    # Calculate document frequency
    doc_freq = Counter()
    for tokens in tokenized_sentences:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    # Calculate TF-IDF scores
    num_docs = len(tokenized_sentences)
    tfidf_scores = {}
    
    for tokens in tokenized_sentences:
        token_freq = Counter(tokens)
        for token, freq in token_freq.items():
            tf = freq / len(tokens) if len(tokens) > 0 else 0  # Term frequency
            idf = math.log((num_docs + 1) / (doc_freq[token] + 1))  # Inverse doc frequency
            tfidf = tf * idf
            
            if token not in tfidf_scores or tfidf > tfidf_scores[token]:
                tfidf_scores[token] = tfidf

    # Return top N keywords
    top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [word for word, score in top_keywords]


def extract_key_information(extractive_sentences, top_keywords):
    """Extract nouns, verbs, adjectives, entities and relationships using POS/NER"""
    docs = [nlp(sent) for sent in extractive_sentences]

    important_nouns = []
    relevant_verbs = []
    key_adjectives = []
    named_entities = []
    noun_verb_pairs = []

    for doc in docs:
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LAW', 'NORP', 'FAC']:
                named_entities.append({'text': ent.text, 'label': ent.label_})

        # Extract nouns, verbs, adjectives using POS tagging
        for token in doc:
            token_lower = token.text.lower()

            if token.pos_ in ['NOUN', 'PROPN']:  # Nouns
                if token_lower in top_keywords or token.dep_ in ['nsubj', 'dobj', 'pobj', 'nsubjpass']:
                    important_nouns.append({
                        'text': token.text,
                        'lemma': token.lemma_,
                        'dep': token.dep_
                    })

            elif token.pos_ == 'VERB':  # Verbs
                if token_lower in top_keywords or token.dep_ in ['ROOT', 'acl', 'relcl']:
                    relevant_verbs.append({
                        'text': token.text,
                        'lemma': token.lemma_,
                        'tense': token.morph.get('Tense')
                    })

            elif token.pos_ in ['ADJ', 'ADV']:  # Adjectives/Adverbs
                if token_lower in top_keywords:
                    key_adjectives.append({
                        'text': token.text,
                        'pos': token.pos_
                    })

        # Extract subject-verb-object relationships
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                objects = [child for child in token.children if child.dep_ in ['dobj', 'pobj']]
                
                for subj in subjects:
                    noun_verb_pairs.append({
                        'subject': subj.text,
                        'verb': token.text,
                        'objects': [obj.text for obj in objects]
                    })

    return {
        'nouns': important_nouns,
        'verbs': relevant_verbs,
        'adjectives': key_adjectives,
        'entities': named_entities,
        'relationships': noun_verb_pairs
    }


def merge_redundant_information(key_info):
    """Remove duplicates and merge overlapping information"""
    # Deduplicate nouns by lemma
    unique_nouns = {}
    for noun in key_info['nouns']:
        lemma = noun.get('lemma', noun.get('text'))
        if lemma not in unique_nouns:
            unique_nouns[lemma] = noun

    # Deduplicate verbs by lemma
    unique_verbs = {}
    for verb in key_info['verbs']:
        lemma = verb.get('lemma', verb.get('text'))
        if lemma not in unique_verbs:
            unique_verbs[lemma] = verb

    # Deduplicate entities by text
    unique_entities = {}
    for entity in key_info['entities']:
        if entity['text'] not in unique_entities:
            unique_entities[entity['text']] = entity

    # Merge relationships with same subject
    merged_relationships = {}
    for rel in key_info['relationships']:
        subj = rel.get('subject', '').strip()
        
        if not subj:
            continue
        
        if subj in merged_relationships:
            merged_relationships[subj]['objects'].extend([
                o for o in rel.get('objects', []) 
                if o not in merged_relationships[subj]['objects']
            ])
        else:
            merged_relationships[subj] = {
                'verb': rel.get('verb', ''),
                'objects': rel.get('objects', [])
            }

    return {
        'nouns': list(unique_nouns.values()),
        'verbs': list(unique_verbs.values()),
        'adjectives': key_info.get('adjectives', []),
        'entities': list(unique_entities.values()),
        'relationships': merged_relationships
    }


def postprocess_summary_text(text):
    """Clean and refine generated summary text"""
    if not text:
        return text
    
    # Fix punctuation and spacing
    text = text.replace('..', '.').replace(' ,', ',').replace(' .', '.')
    
    # Remove duplicate sentences
    parts = [p.strip() for p in text.split('.') if p.strip()]
    seen = set()
    filtered = []
    
    for p in parts:
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        filtered.append(p)
    
    if not filtered:
        return text.strip()
    
    # Rejoin with proper punctuation
    final = '. '.join(filtered)
    if not final.endswith('.'):
        final += '.'
    
    # Capitalize first letter
    final = final[0].upper() + final[1:]
    
    return final


# Avoid generic verbs
MEANINGLESS_VERBS = set(['be', 'go', 'like', 'do', 'have', 'get', 'make', 'say', 'see', 'take', 'put'])
# Connector words for sentences
CONNECTORS = ["highlighting", "emphasizing", "indicating", "addressing", "noting", "concerning", "regarding"]

def generate_abstract_text(merged_info, extractive_sentences):
    """Generate fluent summary from merged information"""
    summary_parts = []

    # Get top elements
    entities = merged_info.get('entities', [])[:3]
    entity_texts = [ent['text'] for ent in entities]

    verbs = merged_info.get('verbs', [])[:5]
    verb_texts = [v['text'] for v in verbs]

    nouns = merged_info.get('nouns', [])[:7]
    noun_texts = [n['text'] for n in nouns]

    relationships = merged_info.get('relationships', {})

    # Build sentences from subject-verb-object relationships
    if relationships:
        rel_items = list(relationships.items())[:4]
        
        for subject, info in rel_items:
            verb = info.get('verb', '').strip()
            
            if not verb or verb.lower() in MEANINGLESS_VERBS:  # Skip generic verbs
                verb = (verb_texts[0] if verb_texts else verb).strip()
            
            objects = [o for o in info.get('objects', []) if o and o.lower() != subject.lower()]
            objects_part = ", ".join(objects[:3]) if objects else ""
            
            topic = noun_texts[0] if noun_texts else (entity_texts[0] if entity_texts else "")
            
            # Create sentence with template
            if objects_part:
                sentence = f"{subject} {verb} {objects_part}"
                if topic:
                    connector = random.choice(CONNECTORS)
                    sentence = f"{sentence}, {connector} {topic}"
            else:
                if topic:
                    connector = random.choice(CONNECTORS)
                    sentence = f"{subject} {verb}, {connector} {topic}"
                else:
                    sentence = f"{subject} {verb}"
            
            summary_parts.append(sentence.strip())

    # Add from extractive if not enough content
    if len(summary_parts) < 2 and extractive_sentences:
        for s in extractive_sentences[:2]:
            clause = s.split(';')[0].split(',')[0].strip()
            if clause and clause not in summary_parts:
                summary_parts.append(clause)

    # Fallback if still no content
    if not summary_parts and entity_texts and verb_texts:
        primary_entity = entity_texts[0]
        primary_verb = verb_texts[0]
        context = f" regarding {noun_texts[0]}" if noun_texts else ""
        summary_parts.append(f"{primary_entity} {primary_verb}{context}")

    # Clean up text
    joined = ". ".join([p.rstrip(' .') for p in summary_parts])
    joined = postprocess_summary_text(joined)

    # Ensure summary is not too short
    tokens = word_tokenize(joined)
    if len(tokens) < 6 and extractive_sentences:
        focus = noun_texts[0] if noun_texts else (entity_texts[0] if entity_texts else "")
        if focus:
            joined = f"{joined} {('The matter concerns ' + focus + '.') if not joined.endswith('.') else ''}"
            joined = postprocess_summary_text(joined)

    return joined if joined else (extractive_sentences[0] if extractive_sentences else "Summary unavailable.")


def abstractive_summary(extractive_sentences):
    """STEP 4: Generate abstractive summary using TF-IDF, POS, NER and text generation"""
    if not extractive_sentences:
        return "No summary available."

    # Step 1: Extract keywords using TF-IDF
    print("\n🔍 Step 1: Calculating TF-IDF for keyword extraction...")
    top_keywords = calculate_tfidf(extractive_sentences, top_n=15)
    print(f"   Top keywords: {', '.join(top_keywords[:10])}")

    # Step 2: Extract key info using POS tagging and NER
    print("\n🔍 Step 2: Extracting key information (POS + NER)...")
    key_info = extract_key_information(extractive_sentences, top_keywords)
    print(f"   - Nouns extracted: {len(key_info['nouns'])}")
    print(f"   - Verbs extracted: {len(key_info['verbs'])}")
    print(f"   - Entities extracted: {len(key_info['entities'])}")
    print(f"   - Relationships found: {len(key_info['relationships'])}")

    # Step 3: Merge redundant information
    print("\n🔍 Step 3: Merging redundant information...")
    merged_info = merge_redundant_information(key_info)
    print(f"   - Unique nouns: {len(merged_info['nouns'])}")
    print(f"   - Unique verbs: {len(merged_info['verbs'])}")
    print(f"   - Unique entities: {len(merged_info['entities'])}")

    # Step 4: Generate abstractive text
    print("\n🔍 Step 4: Generating abstractive summary...")
    abstract_text = generate_abstract_text(merged_info, extractive_sentences)

    return abstract_text


# =====================================================
# MAIN TRAINING PIPELINE (STEP 2: Skip-Gram Training)
# =====================================================
def train_skipgram_from_json(preprocessed_file, log_file='training_log.csv'):
    """
    Complete training pipeline for Skip-Gram word embeddings.
    
    This implements Assignment Step 2: Skip-Gram Training
    
    Args:
        preprocessed_file: Path to JSON file with preprocessed data
                          (created by preprocess_and_save.py)
        log_file: Path to CSV file for logging training results
    
    Returns:
        best_model_path: Path to the best performing trained model
    
    Pipeline Steps:
    1. Load preprocessed data (vocabulary, word mappings, skip-gram pairs)
    2. Validate data integrity
    3. Train model(s) with different hyperparameters
    4. Log all training results
    5. Select and return best model based on lowest loss
    
    Hyperparameters (configurable):
    - Learning rates: [0.3] (can add more for grid search)
    - Batch sizes: [16] (can add more like [16, 32, 64])
    - Epochs: 5
    - Embedding dimension: 50
    """
    # ============================================================
    # LOAD AND VALIDATE PREPROCESSED DATA
    # ============================================================
    print("📂 Loading preprocessed data...")
    with open(preprocessed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract vocabulary size and statistics
    # Handle both new format (with 'stats' key) and old format
    if 'stats' in data:
        # New format: stats are in a separate dictionary
        stats = data['stats']
        vocab_size = stats.get('vocab_size', len(data.get('vocab', [])))
        total_tokens = stats.get('total_tokens', 0)
        unique_words = stats.get('unique_words', 0)
    else:
        # Old format: stats are in root dictionary
        print("⚠️  'stats' key not found. Computing from available data...")
        vocab_size = data.get('vocab_size', len(data.get('vocab', [])))
        if vocab_size == 0 and 'word2idx' in data:
            vocab_size = len(data['word2idx'])
        total_tokens = data.get('total_tokens', 0)
        unique_words = data.get('unique_words', vocab_size)

    # Validate vocabulary size
    if vocab_size == 0:
        raise ValueError(
            "❌ Cannot determine vocabulary size from preprocessed data. "
            "Please regenerate the preprocessed_data.json file using preprocess_and_save.py"
        )

    # Extract word-to-index mapping
    word2idx = data.get('word2idx', {})
    if not word2idx:
        raise ValueError("❌ 'word2idx' mapping not found in preprocessed data!")

    # Extract skip-gram training samples (target, context) pairs
    samples = data.get('samples', [])
    if not samples:
        raise ValueError("❌ No skip-gram samples found in preprocessed data!")

    # Convert samples to tuples (required format)
    samples = [tuple(pair) for pair in samples]

    # Display data statistics
    print(f"📊 Vocabulary size: {vocab_size}")
    print(f"📄 Total tokens: {total_tokens}, Unique words: {unique_words}")
    print(f"🧩 Total skip-gram pairs: {len(samples)}")

    # ============================================================
    # CONFIGURE HYPERPARAMETERS
    # ============================================================
    # Can add more values for grid search (e.g., [0.1, 0.3, 0.5])
    learning_rates = [0.3]
    batch_sizes = [16]
    epochs = 5
    embedding_dim = 50

    # ============================================================
    # TRAINING LOOP (try different hyperparameter combinations)
    # ============================================================
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n🚀 Training with lr={lr}, batch_size={bs}")
            
            # Initialize new model
            model = SkipGramModel(vocab_size, embedding_dim)

            # Train model and track time
            start_time = time.time()
            model.train(samples, epochs=epochs, learning_rate=lr, batch_size=bs)
            end_time = time.time()

            # Save model with hyperparameters in filename
            final_loss = model.last_loss
            model_path = f"skipgram_model_lr{lr}_bs{bs}.pkl"
            model.save(model_path, word2idx)

            # Log training results for comparison
            log_results(
                log_file,
                {
                    'lr': lr,
                    'batch_size': bs,
                    'epochs': epochs,
                    'model_path': model_path
                },
                final_loss,
                end_time - start_time
            )

    # ============================================================
    # SELECT BEST MODEL (lowest loss)
    # ============================================================
    print("\n✅ All training runs completed.")
    df = pd.read_csv(log_file)
    best_row = df.sort_values(by=['loss']).iloc[0]
    print(f"\n🏆 Best Model -> LR={best_row.lr}, Batch={best_row.batch_size}, Loss={best_row.loss}")
    print(f"📁 Using model: {best_row.model_path}")
    
    return best_row.model_path


# =====================================================
# EVALUATION METRICS (STEP 5: Summary Evaluation)
# =====================================================

def calculate_rouge_scores(generated_summary, reference_text):
    """
    Calculate ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.
    
    ROUGE measures n-gram overlap between generated and reference summaries.
    Widely used metric for evaluating automatic summarization.
    
    Args:
        generated_summary: The abstractive summary produced by our system
        reference_text: The reference text (extractive summary or ground truth)
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        Each contains: precision, recall, and F1-score
    
    ROUGE Variants:
    - ROUGE-1: Unigram (single word) overlap
      Measures content coverage
    
    - ROUGE-2: Bigram (two consecutive words) overlap
      Measures fluency and phrase matching
    
    - ROUGE-L: Longest Common Subsequence overlap
      Measures sentence-level similarity (order matters)
    
    Metrics Explained:
    - Precision: What % of generated words are in reference?
    - Recall: What % of reference words are in generated?
    - F1-Score: Harmonic mean of precision and recall
    
    Higher scores = better summary quality (closer to reference)
    """
    from collections import Counter

    # Tokenize both summaries to word level
    gen_tokens = word_tokenize(generated_summary.lower())
    ref_tokens = word_tokenize(reference_text.lower())

    # ========== ROUGE-1: Unigram Overlap ==========
    # Count frequency of each word
    gen_unigrams = Counter(gen_tokens)
    ref_unigrams = Counter(ref_tokens)

    # Count overlapping words (intersection)
    overlap_unigrams = sum((gen_unigrams & ref_unigrams).values())
    
    # Calculate precision, recall, F1
    rouge_1_precision = overlap_unigrams / len(gen_tokens) if gen_tokens else 0
    rouge_1_recall = overlap_unigrams / len(ref_tokens) if ref_tokens else 0
    rouge_1_f1 = (2 * rouge_1_precision * rouge_1_recall /
                  (rouge_1_precision + rouge_1_recall)) if (rouge_1_precision + rouge_1_recall) > 0 else 0

    # ========== ROUGE-2: Bigram Overlap ==========
    # Create bigrams (consecutive word pairs)
    gen_bigrams = Counter(zip(gen_tokens[:-1], gen_tokens[1:]))
    ref_bigrams = Counter(zip(ref_tokens[:-1], ref_tokens[1:]))

    # Count overlapping bigrams
    overlap_bigrams = sum((gen_bigrams & ref_bigrams).values())
    
    # Calculate precision, recall, F1
    rouge_2_precision = overlap_bigrams / max(len(gen_bigrams), 1)
    rouge_2_recall = overlap_bigrams / max(len(ref_bigrams), 1)
    rouge_2_f1 = (2 * rouge_2_precision * rouge_2_recall /
                  (rouge_2_precision + rouge_2_recall)) if (rouge_2_precision + rouge_2_recall) > 0 else 0

    # ========== ROUGE-L: Longest Common Subsequence ==========
    def lcs_length(s1, s2):
        """
        Calculate length of Longest Common Subsequence using Dynamic Programming.
        
        LCS finds longest sequence of words that appear in same order in both texts
        (but don't need to be consecutive).
        
        Example:
        s1: "the court dismissed the case"
        s2: "court dismissed case"
        LCS: "court dismissed case" (length 3)
        """
        m, n = len(s1), len(s2)
        # DP table: dp[i][j] = LCS length of s1[0:i] and s2[0:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    # Words match: extend LCS by 1
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    # Words don't match: take max from previous states
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]

    # Calculate LCS length and metrics
    lcs = lcs_length(gen_tokens, ref_tokens)
    rouge_l_precision = lcs / len(gen_tokens) if gen_tokens else 0
    rouge_l_recall = lcs / len(ref_tokens) if ref_tokens else 0
    rouge_l_f1 = (2 * rouge_l_precision * rouge_l_recall /
                  (rouge_l_precision + rouge_l_recall)) if (rouge_l_precision + rouge_l_recall) > 0 else 0

    # Return all ROUGE scores
    return {
        'ROUGE-1': {'precision': rouge_1_precision, 'recall': rouge_1_recall, 'f1': rouge_1_f1},
        'ROUGE-2': {'precision': rouge_2_precision, 'recall': rouge_2_recall, 'f1': rouge_2_f1},
        'ROUGE-L': {'precision': rouge_l_precision, 'recall': rouge_l_recall, 'f1': rouge_l_f1}
    }


def calculate_bleu_score(generated_summary, reference_text):
    """
    Calculate BLEU (Bilingual Evaluation Understudy) score.
    
    Originally designed for machine translation evaluation,
    also used for summarization quality assessment.
    
    Args:
        generated_summary: System-generated abstractive summary
        reference_text: Reference text (extractive summary or ground truth)
    
    Returns:
        bleu_score: Single score between 0 and 1 (higher is better)
    
    BLEU Components:
    1. N-gram Precision: Measures overlap of n-grams (n=1,2,3,4)
       - 1-gram: individual words
       - 2-gram: consecutive word pairs
       - 3-gram: consecutive word triplets
       - 4-gram: consecutive word quadruplets
    
    2. Geometric Mean: Combines all n-gram precisions
       (Penalizes if any n-gram precision is 0)
    
    3. Brevity Penalty: Penalizes overly short summaries
       (Prevents gaming the metric by generating very short text)
    
    Formula: BLEU = BP × exp(Σ(log(p_n)/N))
    where:
    - BP = brevity penalty
    - p_n = n-gram precision
    - N = number of n-gram levels (4)
    
    Interpretation:
    - 1.0 = Perfect match with reference
    - 0.0 = No overlap with reference
    - Typical good scores: 0.3-0.5 for summarization
    """
    import math
    from collections import Counter

    # Tokenize both texts
    gen_tokens = word_tokenize(generated_summary.lower())
    ref_tokens = word_tokenize(reference_text.lower())

    # Calculate n-gram precisions for n=1,2,3,4
    precisions = []
    for n in range(1, 5):
        # Generate all n-grams from generated summary
        gen_ngrams = Counter(tuple(gen_tokens[i:i + n]) 
                           for i in range(len(gen_tokens) - n + 1))
        
        # Generate all n-grams from reference
        ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) 
                           for i in range(len(ref_tokens) - n + 1))

        # Count overlapping n-grams
        overlap = sum((gen_ngrams & ref_ngrams).values())
        total = sum(gen_ngrams.values())
        
        # Calculate precision for this n-gram level
        precision = overlap / total if total > 0 else 0
        precisions.append(precision)

    # Calculate geometric mean of all n-gram precisions
    # (All precisions must be > 0, otherwise geometric mean = 0)
    if all(p > 0 for p in precisions):
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geo_mean = 0

    # Calculate brevity penalty (BP)
    # Penalizes summaries that are shorter than reference
    gen_len = len(gen_tokens)
    ref_len = len(ref_tokens)
    
    if gen_len >= ref_len:
        brevity_penalty = 1.0  # No penalty if generated is longer
    elif gen_len > 0:
        brevity_penalty = math.exp(1 - ref_len / gen_len)  # Exponential penalty
    else:
        brevity_penalty = 0  # Maximum penalty for empty summary

    # Final BLEU score = BP × geometric mean
    bleu = brevity_penalty * geo_mean
    return bleu


def evaluate_summary_quality(abstractive_summary, extractive_sentences):
    """
    Comprehensive evaluation of generated abstractive summary.
    
    This implements Assignment Step 5: Evaluation
    
    Args:
        abstractive_summary: Generated abstractive summary text
        extractive_sentences: Original extractive summary (used as reference)
    
    Returns:
        Dictionary with all evaluation metrics
    
    Evaluation Metrics:
    1. ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)
       - Measures n-gram overlap with reference
       - Industry standard for summarization evaluation
    
    2. BLEU Score
       - Measures overall quality
       - Originally from machine translation
    
    3. Compression Ratio
       - Shows how much the summary was compressed
       - Lower ratio = more concise summary
    
    Purpose: Objectively measure summary quality and compare different approaches
    """
    # Use extractive summary as reference (join into single text)
    reference_text = " ".join(extractive_sentences)

    print("\n📊 ========== EVALUATION METRICS ==========")
    
    # Calculate ROUGE scores (n-gram overlap metrics)
    rouge_scores = calculate_rouge_scores(abstractive_summary, reference_text)
    print("\n🔹 ROUGE Scores:")
    for metric, scores in rouge_scores.items():
        print(f"   {metric}:")
        print(f"      Precision: {scores['precision']:.4f}")
        print(f"      Recall:    {scores['recall']:.4f}")
        print(f"      F1-Score:  {scores['f1']:.4f}")

    # Calculate BLEU score (overall quality metric)
    bleu = calculate_bleu_score(abstractive_summary, reference_text)
    print(f"\n🔹 BLEU Score: {bleu:.4f}")

    # Calculate summary statistics
    gen_len = len(word_tokenize(abstractive_summary))
    ref_len = len(word_tokenize(reference_text))
    compression_ratio = gen_len / ref_len if ref_len > 0 else 0

    print(f"\n🔹 Summary Statistics:")
    print(f"   Generated length:    {gen_len} words")
    print(f"   Reference length:    {ref_len} words")
    print(f"   Compression ratio:   {compression_ratio:.2%}")

    print("\n==========================================\n")
    
    # Return all metrics in a structured format
    return {
        'rouge': rouge_scores,
        'bleu': bleu,
        'compression_ratio': compression_ratio
    }


# =====================================================
# COMPLETE SUMMARIZATION PIPELINE
# =====================================================
def summarize_with_model(jsonl_file, model_path='skipgram_model.pkl', k=3):
    """
    Complete end-to-end summarization pipeline.
    
    Executes all assignment steps in sequence:
    - Step 3: Extractive Summarization
    - Step 4: Abstractive Summarization
    - Step 5: Evaluation
    
    Args:
        jsonl_file: Path to JSONL file with sentences to summarize
        model_path: Path to trained Skip-Gram model (.pkl file)
        k: Number of sentences to extract (default: 3)
    
    Returns:
        Dictionary containing:
        - extractive: List of selected sentences
        - abstractive: Generated summary text
        - evaluation: All evaluation metrics
    
    Usage Example:
        results = summarize_with_model('test.jsonl', 'model.pkl', k=5)
        print(results['abstractive'])
    """
    # ============================================================
    # LOAD MODEL AND DATA
    # ============================================================
    print("📂 Loading trained model...")
    model, word2idx = SkipGramModel.load(model_path)

    print("📂 Loading text data for summarization...")
    sentences = load_data(jsonl_file)
    print(f"   Loaded {len(sentences)} sentences from {jsonl_file}")

    # ============================================================
    # STEP 3: EXTRACTIVE SUMMARIZATION
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTIVE SUMMARIZATION")
    print("=" * 60)
    # Select top-K most representative sentences using word embeddings
    extractive = extractive_summary(sentences, word2idx, model, k=k)
    print("\n📄 Extractive Summary (Top {} sentences):".format(k))
    for i, s in enumerate(extractive, 1):
        print(f"   {i}. {s}")

    # ============================================================
    # STEP 4: ABSTRACTIVE SUMMARIZATION
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: ABSTRACTIVE SUMMARIZATION")
    print("=" * 60)
    # Generate rephrased, concise summary using NLP techniques
    abstract = abstractive_summary(extractive)
    print("\n📝 Abstractive Summary:")
    print(f"   {abstract}")

    # ============================================================
    # STEP 5: EVALUATION
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATION")
    print("=" * 60)
    # Evaluate summary quality using ROUGE, BLEU metrics
    evaluation_results = evaluate_summary_quality(abstract, extractive)

    # Return all results
    return {
        'extractive': extractive,
        'abstractive': abstract,
        'evaluation': evaluation_results
    }


# =====================================================
# ENTRY POINT - COMMAND-LINE INTERFACE
# =====================================================
if __name__ == "__main__":
    """
    Main entry point with three execution modes:
    
    MODE 1: Training Only
    Usage: python train_skipgram_with_minibatch.py train
    - Trains Skip-Gram model from preprocessed_data.json
    - Saves trained model(s) to disk
    - Logs training results to CSV
    - Selects best model based on lowest loss
    
    MODE 2: Summarization Only
    Usage: python train_skipgram_with_minibatch.py summarize <model_path> [test_file] [k]
    - Loads pre-trained model
    - Generates extractive and abstractive summaries
    - Evaluates summary quality (ROUGE, BLEU)
    - Saves results to timestamped JSON file
    
    MODE 3: Full Pipeline
    Usage: python train_skipgram_with_minibatch.py both [test_file] [k]
    - Trains model from scratch
    - Immediately runs summarization with best model
    - Useful for end-to-end testing
    
    DEFAULT: If no mode specified, runs training only
    
    Examples:
    - python train_skipgram_with_minibatch.py train
    - python train_skipgram_with_minibatch.py summarize model.pkl test.jsonl 5
    - python train_skipgram_with_minibatch.py both test.jsonl 3
    """

    # Check if user provided command-line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        # ============================================================
        # MODE 1: TRAINING ONLY
        # ============================================================
        if mode == 'train':
            print("🚀 Starting training mode...")
            # Train model and get path to best model
            best_model = train_skipgram_from_json("preprocessed_data.json")
            print(f"\n✅ Training complete! Best model: {best_model}")

        # ============================================================
        # MODE 2: SUMMARIZATION ONLY
        # ============================================================
        elif mode == 'summarize':
            # Validate arguments
            if len(sys.argv) < 3:
                print("❌ Usage: python train_skipgram_with_minibatch.py summarize <model_path> [test_file] [k]")
                print("   Example: python train_skipgram_with_minibatch.py summarize skipgram_model_lr0.3_bs16.pkl test.jsonl 5")
                sys.exit(1)

            # Parse command-line arguments
            model_path = sys.argv[2]                                    # Required
            test_file = sys.argv[3] if len(sys.argv) > 3 else 'test.jsonl'  # Optional (default: test.jsonl)
            k = int(sys.argv[4]) if len(sys.argv) > 4 else 3               # Optional (default: 3)

            print(f"🚀 Starting summarization mode...")
            print(f"   Model: {model_path}")
            print(f"   Test file: {test_file}")
            print(f"   Top-K sentences: {k}")

            # Run complete summarization pipeline
            results = summarize_with_model(test_file, model_path, k=k)

            # Save results to JSON file with timestamp
            output_file = f"summary_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'extractive': results['extractive'],
                    'abstractive': results['abstractive'],
                    'evaluation': {
                        # Convert ROUGE scores to serializable format
                        'rouge': {k: {
                            'precision': v['precision'],
                            'recall': v['recall'],
                            'f1': v['f1']
                        } for k, v in results['evaluation']['rouge'].items()},
                        'bleu': results['evaluation']['bleu'],
                        'compression_ratio': results['evaluation']['compression_ratio']
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to: {output_file}")

        # ============================================================
        # MODE 3: FULL PIPELINE (Train + Summarize)
        # ============================================================
        elif mode == 'both':
            print("🚀 Running full pipeline (train + summarize)...")
            
            # Step 1: Train model
            best_model = train_skipgram_from_json("preprocessed_data.json")

            # Parse optional arguments for summarization
            test_file = sys.argv[2] if len(sys.argv) > 2 else 'test.jsonl'
            k = int(sys.argv[3]) if len(sys.argv) > 3 else 3

            # Step 2: Run summarization with trained model
            print("\n" + "=" * 60)
            print("Now running summarization with best model...")
            print("=" * 60 + "\n")

            results = summarize_with_model(test_file, best_model, k=k)
            print(f"\n✅ Full pipeline complete!")

        # ============================================================
        # INVALID MODE
        # ============================================================
        else:
            print(f"❌ Unknown mode: {mode}")
            print("   Valid modes: train, summarize, both")
            sys.exit(1)

    # ============================================================
    # DEFAULT: No mode specified, run training
    # ============================================================
    else:
        print("🚀 No mode specified. Running training by default...")
        print("   Use 'python train_skipgram_with_minibatch.py both' to train and summarize")
        print("   Use 'python train_skipgram_with_minibatch.py summarize <model> <test_file>' to only summarize")
        print()
        
        # Train model
        best_model = train_skipgram_from_json("preprocessed_data.json")
        print(f"\n✅ Training complete! Best model: {best_model}")
        
        # Show helpful next step
        print(f"\n💡 To test summarization, run:")
        print(f"   python train_skipgram_with_minibatch.py summarize {best_model} test.jsonl")
