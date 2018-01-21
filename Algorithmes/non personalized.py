import pandas as pd
import operator
import numpy as np
import time
t0 = time.time()




ratings_df = pd.read_table('~/ml-10M100K/ratings.dat', header=None, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies_df = pd.read_table('~/ml-10M100K/movies.dat', header=None, sep='::', names=['movie_id', 'movie_title', 'movie_genre'])
movies_df = pd.concat([movies_df, movies_df.movie_genre.str.get_dummies(sep='|')], axis=1)
#we dont care about the time the rating was given
del ratings_df['timestamp']

#replace movie_id with movie_title for legibility # merger l'id du movie avec son title ratings_df.head()
ratings_df = pd.merge(ratings_df, movies_df, on='movie_id')[['user_id', 'movie_title', 'movie_id','rating']]



#matrice user/film = rating
ratings_mtx_df = ratings_df.pivot_table(values='rating', index='user_id', columns='movie_title')
ratings_mtx_df.fillna(0, inplace=True)

#print ratings_mtx_df.shape[1]
# noms des films
movie_index = ratings_mtx_df.columns
user_index=ratings_mtx_df.index

score_film={}
for i in movie_index:
      score_film[i]=ratings_mtx_df[i].sum()
orted = sorted(score_film.items(),key=operator.itemgetter(1),reverse=True)
print orted[:5]
t1=time.time()
print "Temps d'execution = %d\n" % (t1 - t0)

