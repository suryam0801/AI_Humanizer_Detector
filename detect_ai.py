# detector.py

# Necessary imports
import torch  # PyTorch for deep learning models
import nltk  # Natural Language Toolkit for text processing
import spacy  # SpaCy for NLP tasks
from spellchecker import SpellChecker  # SpellChecker for grammar checking
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # HuggingFace's transformers for GPT-2 model and tokenizer
from collections import Counter  # Counter for counting token occurrences
from joblib import load  # joblib for loading pre-trained models
import pandas as pd  # pandas for data manipulation

# Initialize necessary tools globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model = load('ai_detector_model.joblib')

print(f"Using device: {device}")


# Function to preprocess new text and extract features
def preprocess_text(text):
    return text.replace('\n', ' ')


# Function to calculate the perplexity of the text using GPT-2 model
# Perplexity is a measure of how well a probability model predicts a sample. A lower perplexity indicates better predictive performance.
def calculate_perplexity(text, tokenizer, model, device):
    max_length = 1024
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    if tokens.size(1) > max_length:
        tokens = tokens[:, :max_length]  # Truncate to the max length
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity


# Function to measure the variance in sentence lengths (burstiness) in the text
# Burstiness refers to the variation in sentence lengths. High burstiness means the text has a mix of very short and very long sentences.
def measure_burstiness(text):
    sentences = nltk.sent_tokenize(text)
    lengths = [len(sentence.split()) for sentence in sentences]
    avg_length = sum(lengths) / len(lengths)
    variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    return variance


# Function to analyze the variation in tone by counting unique part-of-speech (POS) tags
# A higher number of unique POS tags indicates greater variation in tone and sentence structure.
def analyze_tone_variation(text):
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    return len(pos_counts)


# Function to check the number of grammar issues in the text
# This is done by counting the number of misspelled words.
def check_grammar(text):
    words = text.split()
    misspelled = spell.unknown(words)
    return len(misspelled)


# Function to detect repetition in the text
# It calculates a repetition score based on the frequency of repeated words.
def detect_repetition(text):
    tokens = nltk.word_tokenize(text)
    token_counts = Counter(tokens)
    repetition_score = sum(
        count for token, count in token_counts.items() if count > 1) / len(tokens)
    return repetition_score


# Function to extract various features from the text
# These features include perplexity, burstiness, tone variation, grammar issues, and repetition score.
def extract_features(text, tokenizer, gpt2_model, device):
    text = preprocess_text(text)
    features = {
        'perplexity': calculate_perplexity(text, tokenizer, gpt2_model, device),
        'burstiness': measure_burstiness(text),
        'tone_variation': analyze_tone_variation(text),
        'grammar_issues': check_grammar(text),
        'repetition_score': detect_repetition(text)
    }
    return features


# Function to predict the probability of the text being AI-generated
def predict(text):
    # Extract features from the new text
    features = extract_features(text, tokenizer, gpt2_model, device)

    # Convert features to DataFrame
    features_df = pd.DataFrame([features])

    # Predict probability using the loaded model
    probabilities = model.predict_proba(features_df)[0]

    # Get probability of being AI-generated
    ai_probability = probabilities[1]

    return ai_probability
