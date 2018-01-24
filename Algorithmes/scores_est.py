# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt


header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_table('/home/ok/Téléchargements/ml-100k/u.data', sep='\t', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

train_data, test_data = cv.train_test_split(df, test_size=0.25)


#matrix user/item trainset


train_data_matrice = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    train_data_matrice[line[1]-1, line[2]-1] = line[3]

test_data_matrice = np.zeros((n_users, n_items))

for line in test_data.itertuples():
    test_data_matrice[int(line[1])-1, int(line[2])-1] = line[3]
user_similarity = pairwise_distances(train_data_matrice, metric='cosine')
item_similarity = pairwise_distances(train_data_matrice.T, metric='cosine')

#calcul des estimation
def predict(score, similarity, type):
    if type == 'user':
        mean_user_rating = score.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (score - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = score.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
item_prediction = predict(train_data_matrice, item_similarity, type='item')
user_prediction = predict(train_data_matrice, user_similarity, type='user')
#définition du MAE et RMSE
def rmse(prediction, testset):
    prediction = prediction[testset.nonzero()].flatten()
    testset = testset[testset.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, testset ))
def mae(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_absolute_error(prediction, ground_truth)
print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrice))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrice))

print 'User-based CF MAE: ' + str(mae(user_prediction, test_data_matrice))
print 'Item-based CF MAE: ' + str(mae(item_prediction, test_data_matrice))
