"""
In this code, we try to vislualize the results of an algorithme by printing and sorting the best 
predictions and the worst ones and comparing their error rate.
"""
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import KNNBasic
import os
from surprise import Reader
import pandas as pd
import time
from collections import defaultdict
t0 = time.time()

train_file = os.path.expanduser('/home/ok/test.dat')
reader = Reader(line_format='user item rating timestamp', sep='::',skip_lines=1)
data = Dataset.load_from_file(train_file,reader=reader)
trainset = data.build_full_trainset()
movies=pd.read_csv("~/ok/movies.csv",names=["movieId","title","genre"],sep=',')

algo = KNNBasic()
testset=trainset.build_anti_testset()

algo.train(trainset)
predictions = algo.test(testset)
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])


def get_Iu(uid):
    """Return the number of items rated by given user
    Args:
        uid: The raw id of the user.
    Returns:
        The number of items rated by the user.
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0


def get_Ui(iid):
    """Return the number of users that have rated given item
    Args:
        iid: The raw id of the item.
    Returns:
        The number of users that have rated the item.
    """
    try:
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:  # item was not part of the trainset
        return 0


df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)

#takes the best 100 rpedictions. and the worst ten predictions.
best_predictions = df.sort_values(by='err')[:100]
worst_predictions = df.sort_values(by='err')[-10:]
print (best_predictions)
print(worst_predictions)
#This function return the 6 nearest nighbors for an item with the iid=1.
v =algo.get_neighbors(1,6)

t1 = time.time()
print "Temps d'execution = %d\n" % (t1 - t0)

