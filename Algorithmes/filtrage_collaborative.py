import pandas as pd
import numpy as np
import time
t0=time.time()

#we will need movies and their ratings, we import our data and define the format.
ratings_df = pd.read_table('~/ml-1m/ratings.dat', header=None, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies_df = pd.read_table('~/ml-1m/movies.dat', header=None, sep='::', names=['movie_id', 'movie_title', 'movie_genre'])
movies_df = pd.concat([movies_df, movies_df.movie_genre.str.get_dummies(sep='|')], axis=1)

# the timestamp of the ratings is not useful for us, we can get rid of it.
del ratings_df['timestamp']

#merge movies ids with the movies titles and adding it to rating_df
ratings_df = pd.merge(ratings_df, movies_df, on='movie_id')[['user_id', 'movie_title', 'movie_id','rating']]


# we construct a matrix of (user/film ) 
#and M(user,film)= rating given to the film by the user
ratings_mtx_df = ratings_df.pivot_table(values='rating', index='user_id', columns='movie_title')
ratings_mtx_df.fillna(0, inplace=True)


# movies's indexs
movie_index = ratings_mtx_df.columns

#we construct here the correlation matrix which is a matrix of mXm dimensions, 
#Mij represents the correlation between i andj.
corr_matrix = np.corrcoef(ratings_mtx_df.T)

#as an example, we take Toy Story film as a movie the user watched and liked.
favoured_movie_title = 'Toy Story (1995)'
#we get his index
favoured_movie_index = list(movie_index).index(favoured_movie_title)
# get the value of correclation for our film.
P = corr_matrix[favoured_movie_index]
#we return only movies with a high correlation with Toy Story

def get_movie_similarity(movie_title):
    '''
        this function return the correlation vector for a movie.
        args:
            movie_title: title of a movie.
        returns:
            vector of the correlation for the movie given in parameters.
    '''
    movie_idx = list(movie_index).index(movie_title)
    return corr_matrix[movie_idx]

def get_movie_recommendations(user_movies):
    '''
        this function returns all the movies sorted by their correlation with the user.
        args:
            user_movies: list containg the movies that the user watched and liked.
        returns :
            similarities_df: ordered list of movies sorted by their correlation with the user.
    '''
    movie_similarities = np.zeros(corr_matrix.shape[0])
    for movie_id in user_movies:
        movie_similarities = movie_similarities + get_movie_similarity(movie_id)
    similarities_df = pd.DataFrame({
        'movie_title': movie_index,
        'sum_similarity': movie_similarities
        })
    similarities_df = similarities_df[~(similarities_df.movie_title.isin(user_movies))]
    similarities_df = similarities_df.sort_values(by=['sum_similarity'], ascending=False)
    return similarities_df

# we use here an example of a user id for the test.
sample_user = 21
print ratings_df[ratings_df.user_id==sample_user].sort_values(by=['rating'], ascending=False)
#we take the user's movies as a list. then calculate the recommendations.
sample_user_movies = ratings_df[ratings_df.user_id==sample_user].movie_title.tolist()
recommendations = get_movie_recommendations(sample_user_movies)

#and here is the top 20 recommended movies
print recommendations.movie_title.head(20)
t1 = time.time()
print "Temps d'execution = %d\n" % (t1 - t0)