# coding=utf-8
import pandas as pd
import numpy as np
from collections import OrderedDict
import time

t0=time.time() # pour calculer le temps d'execution.

# Dataframe de notre base de données
films_df = pd.read_table('~/ml-10M100K/movies.dat', header=None, sep='::',
	names=['movie_id', 'movie_title', 'movie_genre'])
# on sépare nos données par | pour avoir les categories des films
films_df = pd.concat([films_df, films_df.movie_genre.str.get_dummies(sep='|')], axis=1)

# les categories des films .
film_categories = films_df.columns[3:]

#on construit le vecteur des preference pour un utilisateur :
user_preferences = OrderedDict(zip(film_categories, []))
user_preferences['Action'] = 5
user_preferences['Adventure'] = 5
user_preferences['Animation'] = 1
user_preferences["Children's"] = 1
user_preferences["Comedy"] = 3
user_preferences['Crime'] = 2
user_preferences['Documentary'] = 1
user_preferences['Drama'] = 1
user_preferences['Fantasy'] = 5
user_preferences['Film-Noir'] = 1
user_preferences['Horror'] = 2
user_preferences['Musical'] = 1
user_preferences['Mystery'] = 3
user_preferences['Romance'] = 1
user_preferences['Sci-Fi'] = 5
user_preferences['War'] = 3
user_preferences['Thriller'] = 2
user_preferences['Western'] =1

# produit scalaire
def produit_scalaire(vector_1, vector_2):
    return sum([ i*j for i,j in zip(vector_1, vector_2)])
#le produit entre les categories des films et les preferences d'utilisateurs.
def get_movie_score(movie_features, user_preferences):
    return produit_scalaire(movie_features, user_preferences)
# on prend l'exemple de toy story, qui est la premiere ligne de notre base de données
toy_story_categorie = films_df.loc[0][film_categories]
#print toy_story_categorie
toy_story_user_predicted_score = produit_scalaire(toy_story_categorie, user_preferences.values())

# on liste les recommandation selon les preferences d'utilisateurs.
def get_recommendations(user_preferences, n_recommendations):
    # on ajoute un colonne pour le score calculé pour un film selon l'utilisateur
    films_df['score'] = films_df[film_categories].apply(get_movie_score,
                                                           args=([user_preferences.values()]), axis=1)
    return films_df.sort_values(by=['score'], ascending=False)['movie_title'][:n_recommendations]
#we get the ten first user preferences.
print get_recommendations(user_preferences, 10)
t1 = time.time()
print "Temps d'execution = %d\n" % (t1 - t0)