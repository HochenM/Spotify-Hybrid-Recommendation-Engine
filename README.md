# 🎵 Spotify Content-Based Recommendation Engine

A text-based, content-driven music recommendation system built from scratch using Python, Pandas, and Scikit-Learn. This engine processes raw Spotify track metadata, cleans it using custom regular expressions, and applies Natural Language Processing (NLP) concepts to predict similar tracks.

## 🚀 How It Works
The engine uses a deterministic dual-model layout to calculate track similarities based on a unified "metadata soup":
1. **Custom NLP Tokenization:** Standardizes structural text features. It strategically preserves spaces in track names to maximize word-association matching, while replacing spaces with underscores (`_`) for artists, genres, and albums to enforce exact-token matching.
2. **Text Vectorization:** Converts the combined metadata text string (`combine`) into a mathematical frequency matrix using `CountVectorizer`.
3. **Dual Prediction Engines:** Calculates track similarities simultaneously using **Cosine Similarity** array sorting and **Nearest Neighbors (Brute/Cosine)** to guarantee mathematical alignment and verification.

## 🛠️ Tech Stack & Concepts
* **Language:** Python 3.11
* **Libraries:** Pandas, NumPy, Scikit-Learn (`CountVectorizer`, `NearestNeighbors`, `cosine_similarity`), Regex (`re`)
* **Core Concepts:** Vector Space Models, Content-Based Filtering, Text Preprocessing, Human-Input Validation

## 🔧 Key Engineering Solutions Implemented
* **Robust Text Preprocessing:** Engineered a comprehensive `cleaning()` function that uses Regex to completely strip out structural noise (e.g., `ft.`, `feat.`, `remastered`, `radio edit`) and bracketed punctuation that usually corrupts vector models.
* **Input Validation Loop:** Built a user-facing `input()` interface that sanitizes human typing dynamically, matching it against a pre-cleaned search index to eliminate `IndexError` crashes caused by case-sensitivity or accidental trailing spaces.
* **Algorithmic Guardrails:** Leveraged an `.empty` matrix safety net to cleanly handle instances where a user requests a track outside the scope of the training dataset.

## 📈 What I Learned
* How to transform messy, real-world data columns into a structured "metadata soup" optimized for sparse matrices.
* The explicit math behind text tokenization and how design choices (like utilizing underscores vs. spaces) drastically shift how a vectorizer weighs similarities.
