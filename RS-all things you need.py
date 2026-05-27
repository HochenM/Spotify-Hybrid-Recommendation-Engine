
# Spotify Recommendation System


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv('spotify_data clean.csv')
print(df.shape)
df


# Drop unnecessary raw columns
df.drop(columns=['track_id', 'track_number', 'track_popularity', 'explicit', 'artist_popularity', 'artist_followers',
                 'album_id', 'album_total_tracks', 'album_type', 'track_duration_min'], inplace=True, errors='ignore')


# Cleaning function using regex
import re


def cleaning(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()

    # Remove features and common noise words completely
    noises = [r'\bft\b', r'\bfeat\b', r'\bfeaturing\b',
              r'\bremastered\b', r'\bremaster\b',
              r'\bradio edit\b', r'\bbonus track\b']
    for noise in noises:
        text = re.sub(noise, '', text)

    # Clean up punctuation like (, ), [, ], -, and extra dots
    text = re.sub(r'[\(\)\[\]\-\._,]', ' ', text)

    # Strip out extra white spaces left behind by the cleaning
    text = text.split()
    text = ' '.join(text)
    return text



# Apply text cleaning
df['track_name_clean'] = df['track_name'].apply(cleaning)
df['artist_name_clean'] = df['artist_name'].apply(cleaning)
df['artist_genres_clean'] = df['artist_genres'].apply(cleaning)
df['album_name_clean'] = df['album_name'].apply(cleaning)


# Standarize the date format
df["date_clean"] = df["album_release_date"].fillna('').astype(str).str.replace('-', '_')


# Replace spaces with underscores for non-track columns to maintain strict tokens
df['artist_name_clean'] = df['artist_name_clean'].str.replace(' ', '_')
df['artist_genres_clean'] = df['artist_genres_clean'].str.replace(' ', '_')
df['album_name_clean'] = df['album_name_clean'].str.replace(' ', '_')


# Construct final metadata soup
df['combine'] = (
        df['track_name_clean'] + " " +
        df['artist_name_clean'] + " " +
        df['artist_genres_clean'] + " " +
        df['album_name_clean'] + " " +
        df['date_clean']
)


# Vectorize the metadata soup matrix
vectorizer = CountVectorizer()
dataset_matrix = vectorizer.fit_transform(df['combine'])

print("Vocabulary shape:", vectorizer.get_feature_names_out().shape)

# User Input processing
user_song = input("Same Songs as the song you love: ")
user_song_cleaned = cleaning(user_song)

user_song_row = df[df['track_name_clean'] == user_song_cleaned]

if not user_song_row.empty:
    user_song_index = user_song_row.index[0]
    user_song_text = df['combine'].iloc[user_song_index]
    user_song_vector = vectorizer.transform([user_song_text])

    print(
        f"\nTarget Track Identified: '{df['track_name'].iloc[user_song_index]}' by {df['artist_name'].iloc[user_song_index]}\n")

    n_recommend = 5  # Number of final recommendations to show


    # APPROACH 1: COSINE SIMILARITY

    print("--- Approach 1: Cosine Similarity Results ---")
    similarity = cosine_similarity(user_song_vector, dataset_matrix)
    sorted_similarity = np.argsort(similarity)[0][::-1]

    # Exclude the first item [0] since it's the song itself
    cosine_recommendations = sorted_similarity[1:1 + n_recommend]
    print(df[["track_name", "artist_name"]].iloc[cosine_recommendations])


    # APPROACH 2: NEAREST NEIGHBORS

    print("\n--- Approach 2: Nearest Neighbors (Brute/Cosine) Results ---")
    nn = NearestNeighbors(n_neighbors=n_recommend + 1, algorithm='brute', metric='cosine')
    nn.fit(dataset_matrix)

    distances, indices = nn.kneighbors(user_song_vector)

    # Exclude the first item since it represents the target song item itself
    nn_recommendations = indices[0][1:]
    print(df[["track_name", "artist_name"]].iloc[nn_recommendations])

else:
    print("❌ Song not found. Check spelling or try another song!")