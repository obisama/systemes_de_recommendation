from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import time
from collections import defaultdict
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import numpy as np

"""
This module Contain all the metric we implemented and some that were already implemented in SurpriseLib
**With this Module you can : 
**      _get the top n predictions for an algorithme,
**      _calculate the dcg@k 
**      _calculate the ndcg@k
**      _ Calculate the Mean average precision MAP@k
**      _calculate recision@k and recall@k
**      _calculate the Mae and Rmse
**      _with the multi_metric function, you can calculate all the above by just giving the
**          training data path, and test data path and define the two parameters k and threshold.
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

def dcg_at_k(predictions, k):
        predictions = np.asfarray(predictions)[:k]
        if predictions.size:
            return np.sum(np.subtract(np.power(2, predictions), 1) / np.log2(np.arange(2, predictions.size + 2)))
        return 0.

def ndcg_at_k(predictions, k):
        #ideal dcg score.
        idcg = dcg_at_k(sorted(predictions, reverse=True), k)
        if not idcg:
            return 0.
        return dcg_at_k(predictions, k) / idcg

        # Function to calculate precision and recall
def precision_recall_at_k(predictions, k=10, threshold=4):
    '''Return precision and recall at k metrics for each user.'''
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precisions, recalls


def Mean_Average_precision(predictions, n=10, threshold=3.5):
        """return the Mean average precision value
           args:
                predictions: the result of a prediction algorithm,
                             should take the form as defined in surprise to work.
                n: is the number of result to take into consideration.
                threshold: define the minimum rating to take a document as relevant. default is 3.5 out of 5.
        """
        # let's retrieve the n first recommended items.
        # we initialize a list to store the average precisions values for every user
        average_precisions = list()
        top_n = get_top_n(predictions,n)
        for item in top_n:
            precisions = list()
            for i in range(1, len(top_n[item])):

                if (top_n[item][i][1] >= threshold and top_n[item][i][2] >= threshold):
                    n_rel = sum((true_r >= threshold) for (_, true_r, _, _) in top_n[item][:i])
                    precisions.append(n_rel / i if i != 0 else 1)
            if precisions:
                average_precisions.append(np.mean(precisions))
        return np.mean(average_precisions) if average_precisions else 1

    #le plus important est ce qui vient apres !!.

def multi_metrics_evaluation(training_fpath,test_fpath,algorithme,k=10,threshold=3.5):
        """multi_metrics_evaluation

            This function calculate ( RMSE, MAE, NDCG@k, PRECISION@k, RECALL@k)
            for a training data set and a test data set and using an algorithme given in parameters,

            Args:
                training_fpath : Path to the training data ( csv file ).
                test_fpath: Path to the test data ( csv file )..
                algorithme: the algorithme wich we gonna evaluate,
                            should be given with his intern parameters initialized, otherwise won't work.
                k : NDCG,Precision and Recall depend on the K, number of result to take into consideration.
                threshold: precision and recall depend on the threshold,
                           is the threshold to determine if a prediction is considered
                           to be good.


            Returns:
                A dict containing the score for the four metrics, and execution time , in this form :

                {
                  "K":~,
                  "Threshold":~ ,
                  "Precision@k":~  ,
                  "Recall@k":~,
                  "NDCG@k":~,
                  "MAE":~,
                  "RMSE":~,
                  "Execution_time":~
                }

            """
        start_time = time.time()

        #training data
        file_path_training_data = os.path.expanduser(training_fpath)
        reader = Reader(line_format='user item rating timestamp', sep=',')
        training_data = Dataset.load_from_file(file_path_training_data, reader=reader)
        raw_ratings = training_data.raw_ratings

        #test_data
        file_path_test_data = os.path.expanduser(test_fpath)
        reader = Reader(line_format='user item rating timestamp', sep=',')
        test_data = Dataset.load_from_file(file_path_test_data, reader=reader)
        test_raw_ratings = test_data.raw_ratings

        #you should train your algo and define parameters before give it as a paramter,
        # only the name of algorithme willn't work.
        algo=algorithme

        #Working on training data here, let's call it set A:

        # retrain on the whole set A
        trainset = training_data.build_full_trainset()
        algo.train(trainset)

        # Compute biased accuracy on A
        predictions = algo.test(trainset.build_testset())

        # Compute unbiased accuracy on Test Data, and let's call it B
        ## first we should build the test data to fit surprise form of data
        ## we will do that by the raw ratings we took from the test file

        testset = training_data.construct_testset(test_raw_ratings)  # testset is now the set B

        ## now we test on the data and get the result predictions
        predictions = algo.test(testset)

        ## we will start with calculating NDCG@k
        ### first we need the top N rediction for every user
        ### we use the top n function defined below.
        top_n = get_top_n(predictions, k)
        ### now we calculate ndcg for every user result

        ndcg_ratings = list() # this list will contain all the ndcg scores for every user request
        for uid, user_ratings in top_n.items():
            predicts = list()
            for (_, _, _, r_ndcg) in user_ratings:
                predicts.append(r_ndcg)
            ndcg_ratings.append(ndcg_at_k(predicts, k))


        # We calculate the Predictions and recalls.and their mean value.
        precisions, recalls = precision_recall_at_k(predictions, k, threshold)
        precision = (sum(prec for prec in precisions.values()) / len(precisions))
        recall = (sum(rec for rec in recalls.values()) / len(recalls))
        print('Unbiased accuracy on Test Set,', end='   ')
        # Now we calculate the MAE and RMSE , of the set B.
        rmse =accuracy.rmse(predictions,False)
        mae = accuracy.mae(predictions,False)
        # ,,,, NDCG mean score.
        ndcg_score = (np.mean(ndcg_ratings))
        map_score=Mean_Average_precision(predictions,k,threshold)
        execution_time=(time.time() - start_time)

        #and we print the results
        print("--- Pour K = : %s" % (k))
        print("--- Threshold = : %s " % threshold)
        print("--- Precision@k : %s . " % (precision))
        print("--- Recall@k : %s . " % (recall))
        print("--- NDCG@K :%s . " % (ndcg_score) )
        print("--- MAP@k : %s" %(map_score))
        print("--- RMSE : %s . " % (rmse))
        print("--- MAE :%s . " % (mae))
        print("--- %s seconds ---" % (execution_time))

        #we return a dict containing all the scores.
        return {"K": k , "Threshold" : threshold , "Precision@k" :precision ,
                "Recall@k": recall,"NDCG@k":ndcg_score,"MAE":mae,"RMSE":rmse,"MAP@k":map_score,
                "Execution_time": execution_time
                }
