# 🎵 Spotify Hybrid Recommendation Engine

A content-based and audio-feature hybrid music recommendation system built using Python and Scikit-Learn.

## 🚀 How it Works
This system uses a dual-engine layout to calculate item similarities:
1. **Metadata Soup (NLP):** Combines cleaned track names, artists, genres, albums, and release years using `CountVectorizer`. 
2. **Audio Performance Vectors (Numerical):** Incorporates raw audio signals (Danceability, Energy, Tempo) using `MinMaxScaler`.
3. **Engines:** Generates final top recommendations utilizing both **Cosine Similarity** and **Nearest Neighbors (Brute/Cosine)** models.

## 🛠️ Tech Stack
* Python 3.11
* Pandas & NumPy
* Scikit-Learn (CountVectorizer, NearestNeighbors, Cosine Similarity)

## 📖 What I Learned / Key Achievements
* Handled human input error by tokenizing artists/genres with underscores (`_`) while preserving spaces in track titles to maximize lexical matching.
* Solved data leakage/crashing by standardizing user inputs via Regex before matrix transformation.
