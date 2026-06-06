"""
=============================================================================
  Movie Review Sentiment Detector — Model Training Pipeline
  SAIA 2163 Final Project | Theme 3
=============================================================================
  This script:
    1. Downloads the IMDB Movie Reviews dataset (50k reviews) from HuggingFace
    2. Preprocesses text (cleaning, tokenization, stopword removal, lemmatization)
    3. Extracts features using TWO methods: TF-IDF and Word2Vec
    4. Trains and compares FOUR models: Naive Bayes, Logistic Regression, SVM, Random Forest
    5. Evaluates models with accuracy, precision, recall, F1-score, confusion matrix
    6. Saves the best models, vectorizers, and evaluation results
=============================================================================
"""

import os
import re
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import Counter

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

# Word2Vec
from gensim.models import Word2Vec

# Dataset
from datasets import load_dataset

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Download & Load Dataset
# ──────────────────────────────────────────────────────────────────────────────

def download_and_prepare_data():
    """Download IMDB 50k Movie Reviews dataset from HuggingFace."""
    csv_path = os.path.join(DATA_DIR, 'imdb_reviews.csv')

    if os.path.exists(csv_path):
        print("✅ Dataset already exists. Loading from CSV...")
        df = pd.read_csv(csv_path)
    else:
        print("📥 Downloading IMDB Movie Reviews dataset from HuggingFace...")
        dataset = load_dataset("imdb", trust_remote_code=True)

        # Combine train and test splits
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        df = pd.concat([train_df, test_df], ignore_index=True)

        # Map labels: 0 = negative, 1 = positive
        df.columns = ['text', 'label']
        df['sentiment'] = df['label'].map({0: 'negative', 1: 'positive'})

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Dataset saved to {csv_path}")
        print(f"   Total reviews: {len(df):,}")
        print(f"   Positive: {(df['label'] == 1).sum():,}")
        print(f"   Negative: {(df['label'] == 0).sum():,}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Text Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def setup_nltk():
    """Download required NLTK data."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


def preprocess_text(text):
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Remove HTML tags
      3. Remove URLs
      4. Remove special characters & numbers
      5. Tokenize
      6. Remove stopwords
      7. Lemmatize
    """
    # Lowercase
    text = text.lower()

    # Remove HTML tags (common in IMDB data)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens), tokens


def preprocess_dataset(df):
    """Preprocess entire dataset."""
    print("\n🔧 Preprocessing text data...")
    start = time.time()

    processed = df['text'].apply(preprocess_text)
    df['clean_text'] = processed.apply(lambda x: x[0])
    df['tokens'] = processed.apply(lambda x: x[1])

    elapsed = time.time() - start
    print(f"   ✅ Preprocessing complete in {elapsed:.1f}s")
    print(f"   Sample cleaned text: {df['clean_text'].iloc[0][:100]}...")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_tfidf_features(X_train_text, X_test_text):
    """Feature Extraction Method 1: TF-IDF with bigrams."""
    print("\n📊 Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigrams + Bigrams
        min_df=5,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    print(f"   ✅ TF-IDF matrix shape: {X_train_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, tfidf


def extract_word2vec_features(X_train_tokens, X_test_tokens):
    """Feature Extraction Method 2: Word2Vec embeddings (average pooling)."""
    print("\n📊 Training Word2Vec model...")

    # Train Word2Vec
    w2v_model = Word2Vec(
        sentences=X_train_tokens.tolist(),
        vector_size=200,
        window=5,
        min_count=3,
        workers=4,
        epochs=10,
        sg=1  # Skip-gram
    )

    def get_doc_vector(tokens, model, size=200):
        """Average Word2Vec vectors for all tokens in a document."""
        vectors = [model.wv[t] for t in tokens if t in model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(size)

    X_train_w2v = np.array([get_doc_vector(t, w2v_model) for t in X_train_tokens])
    X_test_w2v = np.array([get_doc_vector(t, w2v_model) for t in X_test_tokens])

    print(f"   ✅ Word2Vec matrix shape: {X_train_w2v.shape}")
    return X_train_w2v, X_test_w2v, w2v_model


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Model Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_name):
    """Train multiple models and evaluate them."""
    print(f"\n🤖 Training models with {feature_name} features...")

    models = {}

    # --- Model 1: Naive Bayes (only for TF-IDF, requires non-negative) ---
    if feature_name == "TF-IDF":
        print("   Training Naive Bayes...")
        nb = MultinomialNB(alpha=0.1)
        nb.fit(X_train, y_train)
        models['Naive Bayes'] = nb

    # --- Model 2: Logistic Regression ---
    print("   Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # --- Model 3: SVM ---
    print("   Training SVM (Linear)...")
    svm_base = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    # Wrap in CalibratedClassifierCV for probability estimates
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_train, y_train)
    models['SVM'] = svm

    # --- Model 4: Random Forest ---
    print("   Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # Evaluate all models
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        print(f"   ✅ {name:25s} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return models, results


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Save Everything
# ──────────────────────────────────────────────────────────────────────────────

def save_artifacts(models_tfidf, models_w2v, results_tfidf, results_w2v,
                   tfidf_vectorizer, w2v_model, df):
    """Save models, vectorizers, and evaluation results."""
    print("\n💾 Saving artifacts...")

    # Save TF-IDF vectorizer
    with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    # Save Word2Vec model
    w2v_model.save(os.path.join(MODEL_DIR, 'word2vec_model.model'))

    # Save TF-IDF models
    for name, model in models_tfidf.items():
        fname = f"tfidf_{name.lower().replace(' ', '_')}.pkl"
        with open(os.path.join(MODEL_DIR, fname), 'wb') as f:
            pickle.dump(model, f)

    # Save Word2Vec models
    for name, model in models_w2v.items():
        fname = f"w2v_{name.lower().replace(' ', '_')}.pkl"
        with open(os.path.join(MODEL_DIR, fname), 'wb') as f:
            pickle.dump(model, f)

    # Combine all results
    all_results = {}
    for name, res in results_tfidf.items():
        all_results[f"TF-IDF + {name}"] = res
    for name, res in results_w2v.items():
        all_results[f"Word2Vec + {name}"] = res

    # Save evaluation results
    with open(os.path.join(MODEL_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Find and save best model name
    best_model_name = max(all_results, key=lambda k: all_results[k]['f1_score'])
    best_info = {
        'best_model': best_model_name,
        'f1_score': all_results[best_model_name]['f1_score'],
        'accuracy': all_results[best_model_name]['accuracy']
    }
    with open(os.path.join(MODEL_DIR, 'best_model_info.json'), 'w') as f:
        json.dump(best_info, f, indent=2)

    # Save dataset statistics
    stats = {
        'total_reviews': len(df),
        'positive_reviews': int((df['label'] == 1).sum()),
        'negative_reviews': int((df['label'] == 0).sum()),
        'avg_review_length': float(df['text'].str.len().mean()),
        'avg_word_count': float(df['text'].str.split().str.len().mean()),
        'avg_clean_word_count': float(df['clean_text'].str.split().str.len().mean()),
    }

    # Top words for positive and negative
    pos_tokens = df[df['label'] == 1]['tokens'].explode()
    neg_tokens = df[df['label'] == 0]['tokens'].explode()
    stats['top_positive_words'] = dict(Counter(pos_tokens).most_common(30))
    stats['top_negative_words'] = dict(Counter(neg_tokens).most_common(30))
    stats['top_all_words'] = dict(Counter(df['tokens'].explode()).most_common(50))

    with open(os.path.join(MODEL_DIR, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Save a small sample CSV for display in the app
    sample = df[['text', 'label', 'sentiment', 'clean_text']].sample(500, random_state=42)
    sample.to_csv(os.path.join(DATA_DIR, 'sample_reviews.csv'), index=False)

    print("   ✅ All artifacts saved!")
    print(f"   🏆 Best model: {best_model_name} (F1: {best_info['f1_score']:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  🎬 Movie Review Sentiment Detector — Training Pipeline")
    print("=" * 70)

    start_time = time.time()

    # 1. Download & load data
    df = download_and_prepare_data()

    # 2. Setup NLTK & Preprocess
    setup_nltk()
    df = preprocess_dataset(df)

    # 3. Train-test split (80-20)
    print("\n📐 Splitting data (80% train, 20% test)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    X_train_tokens = df.loc[X_train_text.index, 'tokens']
    X_test_tokens = df.loc[X_test_text.index, 'tokens']

    print(f"   Train: {len(X_train_text):,} | Test: {len(X_test_text):,}")

    # 4. Feature extraction
    X_train_tfidf, X_test_tfidf, tfidf = extract_tfidf_features(X_train_text, X_test_text)
    X_train_w2v, X_test_w2v, w2v_model = extract_word2vec_features(X_train_tokens, X_test_tokens)

    # 5. Train & evaluate models
    models_tfidf, results_tfidf = train_and_evaluate_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF"
    )
    models_w2v, results_w2v = train_and_evaluate_models(
        X_train_w2v, X_test_w2v, y_train, y_test, "Word2Vec"
    )

    # 6. Save everything
    save_artifacts(models_tfidf, models_w2v, results_tfidf, results_w2v,
                   tfidf, w2v_model, df)

    elapsed = time.time() - start_time
    print(f"\n✅ Training pipeline complete in {elapsed / 60:.1f} minutes")
    print("=" * 70)


if __name__ == '__main__':
    main()
