import streamlit as st
import pickle
import pandas as pd
import joblib
import numpy as np
with open('movies.pkl', 'rb') as file:
    data = pickle.load(file)
df=pd.DataFrame(data)
movie_list=df['title'].values
st.title('Movie Recommendation System')
option = st.selectbox(
    "What movie would you like recommendations for?",movie_list)

st.write("You selected:", option)
similarity = joblib.load('similarity___.npz', mmap_mode='r')
#similarity = np.float32(similarity)
recommended_movies=[]
def recommend(movie):
    index = df[df['title'] == movie].index[0]
    movie_list = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:11]
    for i in movie_list:
        recommended_movies.append(df.iloc[i[0]].title)
    return recommended_movies
if st.button("Recommend"):
    recommendations=recommend(option)
    for i in recommendations:
        st.write(i)
else:
    st.write(option)