"""
This code is a data manipulation example, can be useful.
"""

import pandas as pd
import numpy as np
import time
t0=time.time()
ratings_df = pd.read_table('~/pfe/2007/2007.csv', header=None, sep=',', names=['user_id', 'movie_id', 'rating', 'date'])
movies_df = pd.read_table('~/ml-10M100K/movies.dat', header=None, sep='::', names=['movie_id', 'movie_title', 'movie_genre'])
movies_df = pd.concat([movies_df, movies_df.movie_genre.str.get_dummies(sep='|')], axis=1)
tags_df=pd.read_table('~/ml-10M100K/tags.dat',skiprows=1,header=None,sep='::',names=['user_id','movie_id','tag','date'])
#we dont care about the time the rating was given
del ratings_df['date']
del tags_df['date']
ratings_df = pd.merge(ratings_df, movies_df, on='movie_id')[['user_id', 'movie_title', 'movie_id','rating']]

#replace movie_id with movie_title for legibility # merger l'id du movie avec son title ratings_df.head()
ss = pd.merge(ratings_df,tags_df, on=['user_id','movie_id'],how='left')
sss = ss.pivot_table(values='tag', index='user_id', columns='movie_id',aggfunc='first')
sss.fillna(0, inplace=True)
print sss.head
