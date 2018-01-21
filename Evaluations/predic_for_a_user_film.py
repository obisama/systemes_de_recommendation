"""
This code gives is to predict ratings for a specified user for a specifed item(film),
all you need is a dataset( check our repo in github, you will find the csv file we used)
, and you will also need raw iid and uid ( you can get them from the csv files
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import time
from surprise import Dataset
from surprise import Reader
from surprise import KNNBaseline


#getting our dataset from file, check the data folder where all the data is stocked for other dataset parts.
file_path_training_data = os.path.expanduser("/home/obisama/Desktop/PFE_PFE/pfe/2006/3.csv")
reader = Reader(line_format='user item rating timestamp', sep=',')
training_data = Dataset.load_from_file(file_path_training_data, reader=reader)
training_data.split(n_folds=5)
# we choose the algorithme
## we define the similarity options for the algo
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)

# retrain on the whole Training dataset
trainset = training_data.build_full_trainset()
algo.train(trainset)

""" this bloc is an example i used to test the function and to have an uid and iid to test with.
# we can query for a specific user (defined by his id(uid)) and specific film defined by (iid)
uid = str(196)
iid = str(302)  # raw item id They are strings
# get a prediction for specific users and items.
prediction = algo.predict(uid, iid, verbose=True)
"""

while True :
    #get uid
    try:
        uid = int(input("enter the uid, -1 to quit. \n"))
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue

    if uid < 0:
        print("See Ya, have a nice day.")
        break
    else:
        print("cool, the uid you choose is %s" %(uid))
        uid=str(uid) # raw user id They are strings, we use ints to verify the format of the id.
        # let's get the iid:
        try:
            iid= int(input("enter the iid,-1 to quit.\n"))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
        if iid < 0:
            print("See Ya, have a nice day.")
            break
        else:
            print("cool, the iid you choose is %s" % (iid))
            iid=str(iid)
            # we predict and <<all is well>>
            start_time = time.time()
            prediction = algo.predict(uid, iid, verbose=True)# note that we can get give hte true rating in parametes, true rating can be retrieved from the test datasets.
            print("--- %s seconds ---" % (time.time() - start_time))
            continue
