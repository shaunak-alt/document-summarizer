# Document Summarizer

An intelligent text summarization tool that supports both **extractive** and **abstractive** approaches.
It combines traditional NLP (TF-IDF, Logistic Regression, spaCy) with modern deep learning (Hugging Face Transformers) to generate concise and meaningful summaries from long documents.

---

## ✨ Features
- **Extractive Summarization** using:
  - TF-IDF with Logistic Regression
  - spaCy sentence processing
- **Abstractive Summarization** using:
  - Hugging Face Transformer model [`sshleifer/distilbart-cnn-12-6`]
- **Evaluation** with ROUGE metrics
- Works with plain text files as input

---

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/shaunak-alt/document-summarizer
cd document-summarizer
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Additional downloads:
```bash
python -m spacy download en_core_web_sm
```

## 🛠 Requirements
* Python 3.8+
* spaCy
* Transformers
* Torch
* NLTK
* scikit-learn
* rouge-score
* numpy

---

## 🚀 Usage

### Extractive Summarization (ML-based)
```bash
python Document-Summarizer.py
```

This script:
* Loads your document
* Preprocesses text
* Applies TF-IDF + Logistic Regression
* Evaluates with ROUGE scores

### Abstractive Summarization (Transformer-based)
```bash
python AbstractiveSummarizer.py
```

* Loads your text file
* Generates summaries using `distilbart-cnn-12-6`

---
## 📂 Project Structure
```
.
├── AbstractiveSummarizer.py   # Transformer-based summarization
├── Document-Summarizer.py     # ML-based extractive summarization
└── README.md                  # Documentation
```

## 📊 Example
#### Input (sample.txt):
> Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals...

#### Output (Abstractive):
> AI refers to intelligence shown by machines, unlike natural human or animal intelligence.
