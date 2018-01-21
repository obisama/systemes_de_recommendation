"""
This code is a usage example of the evaluator module.
we used the svd algorithme, tuned by the grid search functionality to get the best measure
based on the rmse score.
"""

import Evaluator_module as Em
import os
import csv
import time
from collections import defaultdict
from surprise import Dataset
from surprise import SVD
from surprise import accuracy
from surprise import Reader
from surprise import GridSearch
from surprise import KNNBaseline
import numpy as np






training_set_path="/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/1Month/11.csv"
test_set_path="/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/TestData/12.csv"

file_path_training_data = os.path.expanduser(training_set_path)
reader = Reader(line_format='user item rating timestamp', sep=',')
training_data = Dataset.load_from_file(file_path_training_data, reader=reader)


print('Grid Search...')
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=0)
grid_search.evaluate(training_data)
grid_search.best_estimator['RMSE']

dict = Em.multi_metrics_evaluation(training_set_path,test_set_path,grid_search.best_estimator['RMSE'],10,3.5)
print dict