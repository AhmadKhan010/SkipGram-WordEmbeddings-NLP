import json
import numpy as np
import time
import pickle
import csv
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import spacy
from preprocess_and_save import load_data

# =====================================================
# Load Spacy NLP model for performing abstractive summarization later
# (used to extract named entities and verbs for generating short summaries)
# =====================================================
nlp = spacy.load("en_core_web_sm")


# =====================================================
# SKIP-GRAM MODEL CLASS
# =====================================================
class SkipGramModel:
    """
    Implements a simple Skip-Gram neural network for word embedding training.
    The model learns word vectors by predicting context words given a target word.
    """

    def __init__(self, vocab_size, embedding_dim):
        # Initialize vocabulary size and embedding dimensions
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Weight matrices:
        # W1: Input (word) → Hidden layer (embedding)
        # W2: Hidden layer → Output (context word probabilities)
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.10
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.10

        # Store last loss for logging
        self.last_loss = None

    # -------------------------------------------------
    # Softmax Function: converts raw scores into probabilities
    # -------------------------------------------------
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=0)

    # -------------------------------------------------
    # Forward Pass: get prediction for one target word
    # -------------------------------------------------
    def forward(self, target_idx):
        # Step 1: Retrieve embedding vector for the target word
        h = self.W1[target_idx]

        # Step 2: Multiply by W2 to get scores for all context words
        u = np.dot(h, self.W2)

        # Step 3: Apply softmax to get probability distribution over vocabulary
        y_pred = self.softmax(u)

        return y_pred, h

    # -------------------------------------------------
    # Backward Pass: compute gradients and update weights for a mini-batch
    # -------------------------------------------------
    def backward(self, batch, learning_rate=0.01):
        # Initialize gradients to zero
        grad_W1 = np.zeros_like(self.W1)
        grad_W2 = np.zeros_like(self.W2)
        batch_loss = 0.0

        # Loop through each (target, context) pair in batch
        for target, context in batch:
            # Forward pass: predict context probabilities for target
            y_pred, h = self.forward(target)

            # Create one-hot vector for the true context word
            y_true = np.zeros(self.vocab_size)
            y_true[context] = 1.0

            # Compute error between prediction and true distribution
            error = y_pred - y_true

            # Compute gradients for both weight matrices
            grad_W2 += np.outer(h, error)
            grad_W1[target] += np.dot(self.W2, error)

            # Calculate cross-entropy loss
            batch_loss += -np.log(y_pred[context] + 1e-15)

        # Normalize gradients by batch size
        batch_size = len(batch)
        self.W1 -= (learning_rate / batch_size) * grad_W1
        self.W2 -= (learning_rate / batch_size) * grad_W2

        return batch_loss

    # -------------------------------------------------
    # Training Loop (Mini-batch Gradient Descent)
    # -------------------------------------------------
    def train(self, samples, epochs=5, learning_rate=0.01, batch_size=512):
        num_samples = len(samples)
        for epoch in range(epochs):
            # Shuffle samples at each epoch
            np.random.shuffle(samples)
            total_loss = 0.0
            print(f"\n🧠 Starting Epoch {epoch + 1}/{epochs}... Total samples: {num_samples}")

            # Loop through mini-batches
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch = samples[batch_start:batch_end]
                batch_loss = self.backward(batch, learning_rate)
                total_loss += batch_loss

                # Calculate progress percentage
                progress = (batch_end / num_samples) * 100

                # Print progress updates every 5%
                if batch_start == 0 or int(progress) % 5 == 0:
                    print(f"   🔹 Progress: {progress:.1f}% complete", end='\r', flush=True)

            # Compute average loss for epoch
            avg_loss = total_loss / num_samples
            self.last_loss = avg_loss

            print(f"\n✅ Epoch {epoch + 1}/{epochs} completed. Avg Loss: {avg_loss:.6f}")

    # -------------------------------------------------
    # Save Model (weights + word2idx mapping)
    # -------------------------------------------------
    def save(self, filepath, word2idx):
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

    # -------------------------------------------------
    # Load Model from .pkl file
    # -------------------------------------------------
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = SkipGramModel(data['vocab_size'], data['embedding_dim'])
        model.W1 = data['W1']
        model.W2 = data['W2']
        return model, data['word2idx']


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def log_results(log_file, params, final_loss, train_time):
    """
    Logs training parameters and results into a CSV file.
    Each training configuration (lr, batch size, etc.) is stored with loss and time.
    """
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


# -------------------------------------------------
# Get average embedding of a sentence
# -------------------------------------------------
def get_sentence_embedding(sentence, word2idx, model):
    # Convert words to their indices (use <UNK> if word not in vocab)
    tokens = [word2idx.get(w.lower(), word2idx.get('<UNK>', 0)) for w in word_tokenize(sentence)]

    # Get embeddings for all valid tokens
    embeddings = [model.W1[idx] for idx in tokens if 0 <= idx < model.W1.shape[0]]

    # Return mean of embeddings (or zero vector if none)
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.embedding_dim)


# -------------------------------------------------
# Cosine Similarity between two vectors
# -------------------------------------------------
def cosine_similarity(v1, v2):
    if norm(v1) == 0 or norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


# -------------------------------------------------
# Extractive Summary using sentence embeddings
# -------------------------------------------------
def extractive_summary(sentences, word2idx, model, k=3):
    """
    Selects top-k most representative sentences based on cosine similarity
    between sentence embeddings and the overall document embedding.
    """
    sentence_embeddings = [get_sentence_embedding(s, word2idx, model) for s in sentences]
    doc_embedding = np.mean(sentence_embeddings, axis=0)
    similarities = [cosine_similarity(se, doc_embedding) for se in sentence_embeddings]

    # Pick top-k sentences with highest similarity to document vector
    top_k = np.argsort(similarities)[-k:][::-1]
    return [sentences[i] for i in top_k]


# -------------------------------------------------
# Abstractive Summary: generate a small textual summary
# -------------------------------------------------
def abstractive_summary(extractive_sentences):
    """
    Uses named entities and verbs from extractive summary to generate
    a short, rule-based abstractive summary of the judgment.
    """
    docs = [nlp(s) for s in extractive_sentences]
    entities, verbs = set(), set()

    # Collect entities (people, orgs, locations) and verbs
    for doc in docs:
        for token in doc:
            if token.ent_type_ in ['PERSON', 'ORG', 'GPE'] or token.pos_ == 'NOUN':
                entities.add(token.text)
            if token.pos_ == 'VERB':
                verbs.add(token.text)

    # Generate short summary based on verbs
    if 'convicted' in verbs:
        return "The court convicted the accused and upheld the sentencing."
    elif 'dismissed' in verbs:
        return "The court dismissed the appeal and maintained the original judgment."
    else:
        return "The judgment summary indicates the court upheld its decision."


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================
def train_skipgram_from_json(preprocessed_file, log_file='training_log.csv'):
    # ================== LOAD PREPROCESSED DATA ==================
    print("📂 Loading preprocessed data...")
    with open(preprocessed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract vocabulary and metadata
    stats = data['stats']  # contains total_tokens, unique_words, vocab_size
    vocab_size = stats['vocab_size']
    total_tokens = stats['total_tokens']
    unique_words = stats['unique_words']

    # Extract mappings and samples
    word2idx = data['word2idx']
    samples = [tuple(pair) for pair in data['samples']]  # (target, context) pairs

    print(f"📊 Vocabulary size: {vocab_size}")
    print(f"📄 Total tokens: {total_tokens}, Unique words: {unique_words}")
    print(f"🧩 Total skip-gram pairs: {len(samples)}")

    # ================== HYPERPARAMETERS ==================
    learning_rates = [0.3]
    batch_sizes = [16]
    epochs = 5
    embedding_dim = 50

    # ================== TRAINING LOOP ==================
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n🚀 Training with lr={lr}, batch_size={bs}")
            model = SkipGramModel(vocab_size, embedding_dim)

            # Record start time
            start_time = time.time()
            model.train(samples, epochs=epochs, learning_rate=lr, batch_size=bs)
            end_time = time.time()

            # Record final loss and save model
            final_loss = model.last_loss
            model_path = f"skipgram_model_lr{lr}_bs{bs}.pkl"
            model.save(model_path, word2idx)

            # Log results to CSV
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

    # ================== POST-TRAINING SUMMARY ==================
    print("\n✅ All training runs completed.")
    df = pd.read_csv(log_file)
    best_row = df.sort_values(by=['loss']).iloc[0]
    print(f"\n🏆 Best Model -> LR={best_row.lr}, Batch={best_row.batch_size}, Loss={best_row.loss}")
    print(f"📁 Using model: {best_row.model_path}")
    print(best_row.model_path)
    return best_row.model_path


def summarize_with_model(jsonl_file, model_path='skipgram_model.pkl'):
    print("Loading trained model...")
    model, word2idx = SkipGramModel.load(model_path)

    print("Loading text data for summarization...")
    sentences = load_data(jsonl_file)

    print("Generating Extractive Summary...")
    extractive = extractive_summary(sentences, word2idx, model, k=3)
    print("\nExtractive Summary:")
    for s in extractive:
        print("-", s)

    print("\nGenerating Abstractive Summary...")
    abstract = abstractive_summary(extractive)
    print("Abstractive Summary:\n", abstract)

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    # Train using preprocessed_data.json (created earlier)
    best_model = train_skipgram_from_json("preprocessed_data.json")

#    model= "skipgram_model_lr0.3_bs16.pkl"
#    summarize_with_model('test.jsonl', model)
    # Use best model for summarization
#    summarize_with_model('test.jsonl', best_model)
