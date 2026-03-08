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

pickle.dump(movies, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Model created successfully")