import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from nltk.stem.porter import PorterStemmer
import pickle

def convert_cast(dict_input):
    output = []
    counter = 0
    for i in ast.literal_eval(dict_input):
        if counter != 3:
            output.append(i['name'])
            counter += 1
        else:
            break
    return output

def convert_crew(dict_input):
    output = []
    for i in ast.literal_eval(dict_input):
        if i['job'] == 'Director':
            output.append(i['name'])
    return output

def convert(dict_input):
    output = []
    for i in ast.literal_eval(dict_input):
        output.append(i['name'])
    return output

movies = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credit, on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(convert_crew)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] 

new_dataframe = movies[['movie_id','title','tags']]

new_dataframe['tags'] = new_dataframe['tags'].apply(lambda x:" ".join(x))
new_dataframe['tags'] = new_dataframe['tags'].apply(lambda x: x.lower())

cv =  CountVectorizer(max_features = 5000, stop_words = 'english')
vectors = cv.fit_transform(new_dataframe['tags']).toarray()

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_dataframe['tags'] = new_dataframe['tags'].apply(stem)

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_dataframe[new_dataframe['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_dataframe.iloc[i[0]].title)

pickle.dump(new_dataframe.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
