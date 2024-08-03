# README
# This script trains a model to detect AI-generated text.
# Run this on Google Colab or any other environment with the necessary libraries installed.

# !pip install transformers spacy pyspellchecker scikit-learn joblib
# !python - m spacy download en_core_web_sm


# Install required packages
# from google.colab import files
from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from spellchecker import SpellChecker
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import spacy
import nltk
import json

# Import necessary libraries

# Ensure the GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load data
file_path = 'transformed_dataset.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize necessary tools
nltk.download('punkt')
spell = SpellChecker()

# Initialize GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Feature Extraction Functions


def preprocess_text(text):
    return text.replace('\n', ' ')


def calculate_perplexity(text):
    max_length = 1024
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    if tokens.size(1) > max_length:
        tokens = tokens[:, :max_length]  # Truncate to the max length
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity


def measure_burstiness(text):
    sentences = nltk.sent_tokenize(text)
    lengths = [len(sentence.split()) for sentence in sentences]
    avg_length = sum(lengths) / len(lengths)
    variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    return variance


def analyze_tone_variation(text):
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    return len(pos_counts)


def check_grammar(text):
    words = text.split()
    misspelled = spell.unknown(words)
    return len(misspelled)


def detect_repetition(text):
    tokens = nltk.word_tokenize(text)
    token_counts = Counter(tokens)
    repetition_score = sum(
        count for token, count in token_counts.items() if count > 1) / len(tokens)
    return repetition_score


def extract_features(text):
    text = preprocess_text(text)
    features = {
        'perplexity': calculate_perplexity(text),
        'burstiness': measure_burstiness(text),
        'tone_variation': analyze_tone_variation(text),
        'grammar_issues': check_grammar(text),
        'repetition_score': detect_repetition(text)
    }
    return features


# Extract Features from Data
feature_list = []
for item in data:
    features = extract_features(item['text'])
    # 0 for human-written, 1 for AI-generated
    features['label'] = 0 if item['label'] == 'human' else 1
    feature_list.append(features)

df = pd.DataFrame(feature_list)

# Train the Model
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the Dataset and Model
df.to_csv('training_dataset.csv', index=False)
dump(classifier, 'ai_detector_model.joblib')

# Download the dataset and model
files.download('training_dataset.csv')
files.download('ai_detector_model.joblib')
