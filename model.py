import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("dataset/telugu_movies.csv")

movies = movies[['title','genre','overview','director','cast']]

movies['tags'] = movies['genre'] + " " + movies['overview'] + " " + movies['director'] + " " + movies['cast']

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies=pickle.load(open(os.path.join(BASE_DIR, 'movies.pkl'), 'rb'))
similarity=pickle.load(open(os.path.join(BASE_DIR, 'similarity.pkl'), 'rb'))
print("Model created successfully")