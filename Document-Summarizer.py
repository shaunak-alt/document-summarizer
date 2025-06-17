import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer

nltk.download('punkt')

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess(text):
    text = re.sub(r'\s+', ' ', text)
    sentences = nltk.sent_tokenize(text)
    return sentences

def compute_features(sentences):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X, vectorizer

def create_labels(sentences, top_k=3):
    sorted_ids = sorted(range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True)
    labels = [1 if i in sorted_ids[:top_k] else 0 for i in range(len(sentences))]
    return labels

def evaluate_summary(pred_summary, true_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    return scorer.score(true_summary, pred_summary)

# Load and process document
text = load_text(r'C:\Users\shaun\OneDrive\Desktop\aiko.txt')
sentences = preprocess(text)

# Feature extraction
X, vectorizer = compute_features(sentences)
y = create_labels(sentences, top_k=3)

# Train/test split
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, sentences, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict and generate summary
y_pred = clf.predict(X_test)
summary_sentences = [s for s, p in zip(s_test, y_pred) if p == 1]
pred_summary = ' '.join(summary_sentences)

# True summary
true_summary_sentences = [s for s, label in zip(s_test, create_labels(s_test, top_k=3)) if label == 1]
true_summary = ' '.join(true_summary_sentences)

# Output and evaluation
print("\nGenerated Summary:\n", pred_summary)
print("\nTrue Summary:\n", true_summary)
print("\nROUGE Evaluation:\n", evaluate_summary(pred_summary, true_summary))