"""
MAP : Mean average precision metric , is similar to NDCG@ but more reliable on long lists of results.
    in this document we present a way to calculate Map values for a query.
    ## The code bellow is applied on SVD algorithm,.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
from collections import defaultdict

from surprise import Dataset
from surprise import SVD

def Mean_Average_precision(predictions,n=10,threshold=3.5):
    """return the Mean average precision value
       args:
            predictions: the result of a prediction algorithm,
                         should take the form as defined in surprise to work.
            n: is the number of result to take into consideration.
            threshold: define the minimum rating to take a document as relevant. default is 3.5 out of 5.
    """
    # let's retrieve the n first recommended items.
    top_n = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        if est >= threshold:
            top_n[uid].append((true_r, est))

            # Then sort the predictions for each user and retrieve the n highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

    # we initialize a list to store the average precisions values for every user
    average_precisions = list()
    # an item here is a vector of the top_n documents for every user
    for item in top_n:

        precisions = list()
        for i in range(0, len(top_n[item])):
            if (top_n[item][i][0] >= threshold and top_n[item][i][1] >= threshold):
                # we calculate the number of relevant documents.
                n_rel = sum((true_r >= threshold) for (true_r, _) in top_n[item][:i])
                # we divide on i because we are sure that every document is recommended.
                precisions.append(n_rel / i if i!=0 else 1)
        if precisions:
            average_precisions.append(np.mean(precisions))
    return np.mean(average_precisions)

data = Dataset.load_builtin('ml-100k')

algo = SVD()
trainset = data.build_full_trainset()
algo.train(trainset)

predictions = algo.test(trainset.build_testset())

print("###### MAP")
print(Mean_Average_precision(predictions,10,4))
#0.971740229111