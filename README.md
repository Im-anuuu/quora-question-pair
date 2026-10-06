# Quora Question Pair Similarity

Predicts whether two questions are duplicates using NLP features and a Random Forest.

## Features
- TF-IDF cosine similarity
- Fuzzy matching scores (ratio, partial, token sort, token set)
- Length and common-word features

## Run
pip install -r requirements.txt
python train.py
python app.py

## Results
Accuracy: XX% | F1: XX%