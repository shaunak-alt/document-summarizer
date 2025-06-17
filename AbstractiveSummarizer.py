import spacy
from transformers import pipeline

# Load spaCy model for extractive summarization
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face pipeline for abstractive summarization
abstractive_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extractive_summary(text, max_sentences=3):
    doc = nlp(text)
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in nlp.Defaults.stop_words and word.is_alpha:
            word_frequencies[word.text.lower()] = word_frequencies.get(word.text.lower(), 0) + 1

    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_freq

    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text.lower()]

    # Select top N sentences
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    final_summary = ' '.join([sent.text for sent in summarized_sentences])
    return final_summary

def abstractive_summary(text, max_len=130, min_len=30):
    summary = abstractive_summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

# Example usage
if __name__ == "__main__":
    file_path = r'C:\Users\shaun\OneDrive\Desktop\aiko.txt'
    text = load_text(file_path)

    print("Extractive Summary:\n")
    print(extractive_summary(text))

    print("\nAbstractive Summary:\n")
    print(abstractive_summary(text))
