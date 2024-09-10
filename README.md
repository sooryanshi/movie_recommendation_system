# Movies Recommender System
## About Dataset
### Context
 These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of
 movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters,
 release dates, languages, production companies, countries, TMDB vote counts and vote averages.
 This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a
 scale of 1-5 and have been obtained from the official GroupLens website.
### Acknowledgements
 This dataset is an ensemble of data collected from TMDB and GroupLens.
 The Movie Details, Credits and Keywords have been collected from the TMDB Open API. This product uses the
 TMDb API but is not endorsed or certified by TMDb.
 
  #### This project involves building a movie recommender system using machine learning techniques:
  
### 1. Problem Definition
    Objective: Develop a movie recommender system that suggests movies to users based on their past behavior and preferences.
### 2. Data Collection 
### 3. Data Preprocessing
   ```python
   movies=pd.read_csv('movies_metadata.csv',low_memory=False)
   movies.drop(['adult','budget','homepage','imdb_id','original_language','release_date'	,'revenue'	,'runtime',	'spoken_languages'	,'status',	  
   'tagline','video','popularity','poster_path','production_companies','production_countries','original_title','belongs_to_collection'],axis=1,inplace=True)
   df=movies.copy()
   df.genres=df.genres.apply(ast.literal_eval)
   df=df.explode('genres', ignore_index=False)
   df['genres'] = df['genres'].apply(lambda x: {'id': 0, 'name': np.nan} if pd.isna(x) else x)
   df[['genre_id', 'genre']] =df['genres'].apply(pd.Series)
   df =df.drop(columns=['genres','genre_id'])
   ```
              
### 4. Building the Recommender System
  ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import cosine_similarity as cosim
   tv =TfidfVectorizer(max_features=1800, stop_words='english')
   merged_df['overview'] = merged_df['overview'].fillna('')
   vectors = tv.fit_transform(merged_df['overview']).toarray()
   similarity=np.float32(cosim(vectors))
  ```

### 5. Recommending Movies(Content Based)
    ```python
    def recommend(Movie):
     index=merged_df[merged_df['title']==Movie].index[0]
     movie_list=sorted(list(enumerate(similarity[index])),reverse=True, key=lambda x: x[1])[1:10]
     for i in movie_list:
         print(merged_df.loc[i[0],'title'])
    ```
### 6.  Recommending Movies(Genre Based) : (# this is not included in the deployed part of the project, built for additional search method)
     ```python
     #this is not included in the deployed part of the project, built for additional search method
    def bayesian_average(vote_average, vote_count, global_avg, total_count, C=50):
      return (C * global_avg + vote_count * vote_average) / (C + vote_count)
    merged_df['bayesian_average'] = merged_df.apply(lambda row: bayesian_average(
    row['vote_average'],
    row['vote_count'],
    global_avg_rating,
    total_vote_count
    ), axis=1)
    def movie_genre(Genre,n=10):
      sample_df=pd.DataFrame({})
      sample_df['title']=[]
      sample_df['ratings']=[]
      #index_=genre_recommendation[genre_recommendation['genre']==Genre]
      mean=genre_recommendation[genre_recommendation['genre']==Genre]['bayesian_avg_mean']
      genre_df=merged_df[merged_df['genre']==Genre]
      for i in range(len(genre_df)):
          if genre_df.iloc[i]['bayesian_average']>=mean.values:
              sample_df.loc[len(sample_df)]=[genre_df.iloc[i]['title'],genre_df.iloc[i]['bayesian_average']]
    sample_df.sort_values(by='ratings', ascending=False, inplace=True)
    print(sample_df.head(n))
            
    ```
### 7. Deployment
    ```python
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
    
     ```
### 8. Monitoring and Maintenance
    Set up logging and monitoring to track the performance of the recommender system, and
    schedule regular retraining with new data to keep the recommendations relevant.
### 9. Documentation and Reporting
     Maintain comprehensive documentation of the project, including data sources, preprocessing
     steps, model selection, and evaluation results.

## Tools and Technologies

 ● Programming Language: Python

 ● Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, streamlit, joblib

---

## Conclusion

This project has been an incredible learning journey into the world of machine learning. Throughout the development of this movie recommender system, I have gained hands-on experience with key concepts such as data preprocessing, exploratory data analysis, and collaborative filtering. Implementing algorithms like Matrix Factorization (SVD) has deepened my understanding of how machine learning models can be used to derive meaningful insights from data. Additionally, building and deploying this system has provided valuable insights into the practical challenges of working with real-world data and deploying models in a web application. Overall, this project has not only enhanced my technical skills but also ignited a passion for exploring more advanced machine learning techniques. I look forward to applying these skills to future projects and continuing to grow as a machine learning practitioner.

---

