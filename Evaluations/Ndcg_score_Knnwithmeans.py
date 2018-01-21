"""
In this code we calculate the NDCG score for a request done by an algo from our choice, 
in this case , i used a KNNWithMeans algorithme, in two  methods ( user based and product based)
"""

import os
import csv
import time
from collections import defaultdict
from surprise import Dataset
from surprise import SVD
from surprise import accuracy
from surprise import Reader
from surprise import GridSearch
from surprise import KNNWithMeans
import numpy as np


""" in order to calculate ndcg we need top N prediction for every user
	using custom datasets ( our datasets).
"""
def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    r_ndcg=0
    for uid, iid, true_r, est, _ in predictions:
        if true_r >= 4 :
            r_ndcg=3
        elif true_r >= 3:
            r_ndcg=2
        elif true_r >=1 :
            r_ndcg=1
        else:
            r_ndcg=0
        top_n[uid].append((iid,true_r,est,r_ndcg))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
""" first we calculate DCG@k, see the definition in the repport."""
def dcg_at_k(predictions, k):
        predictions = np.asfarray(predictions)[:k]
        if predictions.size:
            return np.sum(np.subtract(np.power(2, predictions), 1) / np.log2(np.arange(2, predictions.size + 2)))
        return 0.

def ndcg_at_k(predictions, k):
		#we get ideal dcg score of the request.
        ideal_dcg = dcg_at_k(sorted(predictions, reverse=True), k)
        # check that the does exist and not null.
        if not ideal_dcg:
            return 0.
        # Calculate the ndcg@k, 
        ## we say that an ndcg@k score is ideal if it is equal to 1.
        return dcg_at_k(predictions, k) / ideal_dcg

start_time = time.time()

# the training data file:

file_path_training_data = os.path.expanduser("/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/1Month/11.csv")
reader = Reader(line_format='user item rating timestamp', sep=',')
training_data = Dataset.load_from_file(file_path_training_data, reader=reader)

# the test data file
## Notice that we use raw_ratings to construct our dataset so it fit surpriseLib's data form.
file_path_test_data = os.path.expanduser("/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/TestData/12.csv")
reader = Reader(line_format='user item rating timestamp', sep=',')
test_data = Dataset.load_from_file(file_path_test_data, reader=reader)
test_raw_ratings = test_data.raw_ratings


#training_data.raw_ratings = raw_ratings
training_data.split(n_folds=5)

# we define the similarity options for the algo
sim_options1 = {'name': 'pearson_baseline', 'user_based': False}
sim_options2 = {'name': 'pearson_baseline', 'user_based': True}

algo1 = KNNWithMeans(sim_options=sim_options1)
algo2 = KNNWithMeans(sim_options=sim_options2)
# retrain on the whole Training dataset
trainset = training_data.build_full_trainset()
algo1.train(trainset)
algo2.train(trainset)

# Compute unbiased accuracy on test datasets
## Algo1 : 
testset = training_data.construct_testset(test_raw_ratings)  
predictions1 = algo1.test(testset)
top_n = get_top_n(predictions1, n=20)
ndcg_ratings = list() # this list will contain all the ndcg scores for every user request
        for uid, user_ratings in top_n.items():
            predicts = list()
        for (_, _, _, r_ndcg) in user_ratings:
                predicts.append(r_ndcg)
            ndcg_ratings.append(ndcg_at_k(predicts, 20))


print('Unbiased accuracy on TestData, For the KNNBaseline product based ', end=' ')
print("--- Pour K = %s : \n")
print("--- NDCG@K :%s . \n" % (np.mean(ndcg_ratings)))
print("--- %s seconds ---" % (time.time() - start_time))

## Algo2: 
testset = training_data.construct_testset(test_raw_ratings)  
predictions2 = algo2.test(testset)
top_n = get_top_n(predictions2, n=20)
ndcg_ratings = list() # this list will contain all the ndcg scores for every user request
        for uid, user_ratings in top_n.items():
            predicts = list()
        for (_, _, _, r_ndcg) in user_ratings:
                predicts.append(r_ndcg)
            ndcg_ratings.append(ndcg_at_k(predicts, 20))


print('Unbiased accuracy on TestData, For the KNNBaseline user based ', end=' ')
print("--- Pour K = 20: \n")
print("--- NDCG@K :%s . \n" % (np.mean(ndcg_ratings)))
print("--- %s seconds ---" % (time.time() - start_time))
