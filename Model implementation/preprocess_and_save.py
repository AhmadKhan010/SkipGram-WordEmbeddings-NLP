import json
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# ==================== SETUP ====================
nltk_packages = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'stopwords': 'corpora/stopwords'
}
for pkg, path in nltk_packages.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

_re_date = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
_re_year = re.compile(r"\b\d{4}\b")
_re_money = re.compile(r"\bRs\.?\s*[\d,]+(?:/\-)?\b", re.IGNORECASE)
_re_reg = re.compile(r"\b[A-Za-z]{1,4}[-]?\d{2,6}(?:[-/][A-Za-z0-9-]+)?\b")

_stop_words = set(stopwords.words('english'))
_punctuations = set(string.punctuation)

# ==================== HELPERS ====================

def load_data(jsonl_file):
    sentences = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentences.extend(data['Sentences'])
    return sentences

def tokenize_and_filter(sentence, keep_patterns=None):
    if keep_patterns is None:
        keep_patterns = [_re_date, _re_year, _re_money, _re_reg]

    raw_tokens = word_tokenize(sentence)
    out_tokens = []
    for tok in raw_tokens:
        tok_lower = tok.lower()
        keep = any(pat.search(tok) for pat in keep_patterns)
        if not keep and all(ch in _punctuations for ch in tok):
            continue
        if tok_lower in _stop_words and not keep:
            continue
        tok_clean = tok_lower.strip()
        if tok_clean:
            out_tokens.append(tok_clean)
    return out_tokens

def build_vocabulary(sentences, min_freq=5):
    tokens = []
    for sent in sentences:
        toks = tokenize_and_filter(sent)
        tokens.extend(toks)

    word_counts = Counter(tokens)
    total_tokens = len(tokens)
    unique_words = len(word_counts)
    print("First 50 filtered tokens:", tokens[:50])
    print("Most common words:", word_counts.most_common(10))
    print("Total unique words:", len(word_counts))
    print("Total tokens:", len(tokens))

    vocab_words = [word for word, count in word_counts.items() if count >= min_freq]
    vocab = ['<PAD>', '<UNK>'] + vocab_words
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word, vocab, total_tokens, unique_words

def generate_skipgram_samples(sentences, word2idx, window_size=2):
    samples = []
    for sentence in sentences:
        token_list = tokenize_and_filter(sentence)
        tokens = [word2idx.get(t, word2idx['<UNK>']) for t in token_list]
        for i, target in enumerate(tokens):
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    samples.append((target, tokens[j]))

    print("First 20 Skip-gram samples (target, context):", samples[:20])
    return samples

def preprocess_and_save(jsonl_file, output_file='preprocessed_data.json'):
    print("Loading data...")
    sentences = load_data(jsonl_file)
    print(f"Loaded {len(sentences)} sentences.")

    print("Building vocabulary...")
    word2idx, idx2word, vocab, total_tokens, unique_words = build_vocabulary(sentences, min_freq=5)
    print(f"Vocabulary size: {len(vocab)}")

    print("Generating skip-gram samples...")
    samples = generate_skipgram_samples(sentences, word2idx, window_size=2)
    print(f"Generated {len(samples)} samples.")

    data = {
        'stats': {
            'total_tokens': total_tokens,
            'unique_words': unique_words,
            'vocab_size': len(vocab)
        },
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab': vocab,
        'samples': samples
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    print(f"\n✅ Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_and_save('1987_documents.jsonl', 'preprocessed_data.json')
