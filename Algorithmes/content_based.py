import pandas as pd
import numpy as np
from collections import OrderedDict
import time
t0=time.time() # to calculate the execution time as it's an important feature.

#we need the movies list
movies_df = pd.read_table('~/ml-10M100K/movies.dat', header=None, sep='::', 
	names=['movie_id', 'movie_title', 'movie_genre'])
movies_df = pd.concat([movies_df, movies_df.movie_genre.str.get_dummies(sep='|')], axis=1)

# we select the categories to define the user preferences.
movie_categories = movies_df.columns[3:]

#we construct an example of a user preferences : 
user_preferences = OrderedDict(zip(movie_categories, []))
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

# the name says all :P
def scalar_product(vector_1, vector_2):
    return sum([ i*j for i,j in zip(vector_1, vector_2)])
#take user preferences and movie features to calulate the product , then returns the result.
def get_movie_score(movie_features, user_preferences):
    return scalar_product(movie_features, user_preferences)
#here we work with the Toy_Story film as an example
toy_story_features = movies_df.loc[0][movie_categories]
toy_story_user_predicted_score = scalar_product(toy_story_features, user_preferences.values())

# get the sorted predictions for the user based on his preferences.
def get_recommendations(user_preferences, n_recommendations):
    #we add a column to the movies_df dataset with the calculated score for each movie for the given user
    movies_df['score'] = movies_df[movie_categories].apply(get_movie_score,
                                                           args=([user_preferences.values()]), axis=1)
    return movies_df.sort_values(by=['score'], ascending=False)['movie_title'][:n_recommendations]
#we get the ten first user preferences.
print get_recommendations(user_preferences, 10)
t1 = time.time()
print "Temps d'execution = %d\n" % (t1 - t0)