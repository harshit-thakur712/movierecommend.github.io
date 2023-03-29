#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[30]:


movies = pd.read_csv("C:\\Users\\320204024\\source\\resume_pro\\movie-recommender-system-tmdb-dataset\\tmdb_5000_movies .csv")
credits = pd.read_csv("C:\\Users\\320204024\\source\\resume_pro\\movie-recommender-system-tmdb-dataset\\tmdb_5000_credits.csv")


# In[31]:


movies.head(2)


# In[4]:


movies.shape


# In[5]:


credits.head()


# In[32]:


movies = movies.merge(credits,on='title')


# In[ ]:


movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[33]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


movies.head()


# In[12]:


import ast


# In[34]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[35]:


movies.dropna(inplace=True)


# In[36]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[37]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[ ]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[38]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[39]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[40]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[41]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[42]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[22]:


#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


# In[43]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[44]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[26]:


movies.head()


# In[45]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[46]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[47]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[48]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    


# In[50]:


vector = cv.fit_transform(new['tags']).toarray()


# In[ ]:


vector.shape


# In[51]:


from sklearn.metrics.pairwise import cosine_similarity


# In[52]:


similarity = cosine_similarity(vector)


# In[ ]:


similarity


# In[ ]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[53]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        
    


# In[54]:


recommend('Gandhi')


# In[55]:


import pickle


# In[56]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




