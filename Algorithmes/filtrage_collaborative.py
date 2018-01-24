# coding=utf-8
import pandas as pd
import numpy as np
import time
t0=time.time()

#on importe le fichier des rates et le fichier Movies ainsi les merger selon l'ID du movie.
ratings_df = pd.read_table('~/ml-1m/ratings.dat', header=None, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])
films_df = pd.read_table('~/ml-1m/movies.dat', header=None, sep='::', names=['movie_id', 'movie_title', 'movie_genre'])
films_df = pd.concat([films_df, films_df.movie_genre.str.get_dummies(sep='|')], axis=1)

# on aura pas besoin du timestamp
del ratings_df['timestamp']


ratings_df = pd.merge(ratings_df, films_df, on='movie_id')[['user_id', 'movie_title', 'movie_id','rating']]


# on construit la matrice User/film=score
#and M(user,film)= rating given to the film by the user
ratings_matrice_df = ratings_df.pivot_table(values='rating', index='user_id', columns='movie_title')
ratings_matrice_df.fillna(0, inplace=True)


# les index des movies
movie_index = ratings_matrice_df.columns

#we construct here the correlation matrix which is a matrix of mXm dimensions, 
#Mij represente la corrélation entre i et j.
corr_matrice = np.corrcoef(ratings_matrice_df.T)

#on prend par exemple le film préferé est Toy story.
film_pref_title = 'Toy Story (1995)'
# l'index du film
film_pref_index = list(movie_index).index(film_pref_title)
# et on cherche sa corrélation avec d'autres films.
P = corr_matrice[film_pref_index]
#on peut retourner les fortes correlation par le print suivant
#print list(movie_index[(P>0.4) & (P<1.0)])

def get_movie_similarity(movie_title):
    '''
        la fonction retourne le vecteur de correlation pour le film donné en paramétre

    '''
    movie_idx = list(movie_index).index(movie_title)
    return corr_matrice[movie_idx]

def get_movie_recommendations(user_movies):
    '''

            similarities_df: liste ordonnée des films triée par la correlation avec l'utilisateur.
    '''
    movie_similarities = np.zeros(corr_matrice.shape[0])
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