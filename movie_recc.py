#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies=pd.read_csv('movies.csv')

movies['Content']= movies['Genre']+' ' + movies['Lead Studio']

converted=TfidfVectorizer(stop_words='english')
converted_matrix=converted.fit_transform(movies['Content'])

cosine_sim=linear_kernel(converted_matrix,converted_matrix)

indices = pd.Series(movies.index, index=movies['Film']).drop_duplicates()

# Recommender function
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['Film'].iloc[movie_indices]

# Try it
print(get_recommendations("Youth in Revolt"))


# In[ ]:




