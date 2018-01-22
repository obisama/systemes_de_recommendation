"""
In this code, we are going to evaluate two algorithms, a Collaboratif filtering and a baseline estimate algorithme using my evaluator module i created applied to data from four
periods to predict results on a futur period ,

Scenario:

        we want to predict the ratings for Spetember 2008, to do so we are gonna use four training data,
         and then test and evaluate for each one.
            case1: we have data of November and we want to predict data of December.
            case2: we have data for : September,October,November and we want tp repdict December data's.
            case3: we have data of the six months before Before Spetember.
            case4: we hava data of one year before September.

        For the evaluation we took, K as in Precision@k, Recall@k and NDCG@k equal to 20.
            we suppose that a result is relevent if it's rating is superior or equal to four.
"""

import Evaluator_module as Em
import os
import csv
import time
from collections import defaultdict
from surprise import Dataset
from surprise import KNNWithMeans
from surprise import Reader

import numpy as np


# we define the path for our test_dataset:
## in our case it's data of December 2008 and is the same for the four scenarios.
test_set_path="/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/TestData/12.csv"


# Case 1 : 1 month
print("-------Case 1: 1month")

training_set_path="/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/1Month/11.csv"
file_path_training_data = os.path.expanduser(training_set_path)
reader = Reader(line_format='user item rating timestamp', sep=',')
training_data = Dataset.load_from_file(file_path_training_data, reader=reader)
sim_options1 = {'name': 'pearson_baseline', 'user_based': False}
sim_options2 = {'name': 'pearson_baseline', 'user_based': True}
algo1 = KNNWithMeans(sim_options=sim_options1)
algo2 = KNNWithMeans(sim_options=sim_options2)
dict1 = Em.multi_metrics_evaluation(training_set_path,test_set_path,algo1)
dict2 = Em.multi_metrics_evaluation(training_set_path,test_set_path,algo2)
print (" dict 1 :")
print(dict1)
print("dict 2 :")
print(dict2)
print("------ End Of Case 1")

# Case 2 : 3 month
print("-------Case 2: 3month")

training_set_path="/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/3Months/3months.csv"
file_path_training_data = os.path.expanduser(training_set_path)
reader = Reader(line_format='user item rating timestamp', sep=',')
training_data = Dataset.load_from_file(file_path_training_data, reader=reader)

sim_options1 = {'name': 'pearson_baseline', 'user_based': False}
sim_options2 = {'name': 'pearson_baseline', 'user_based': True}
algo1 = KNNWithMeans(sim_options=sim_options1)
algo2 = KNNWithMeans(sim_options=sim_options2)
dict1 = Em.multi_metrics_evaluation(training_set_path,test_set_path,algo1)
dict2 = Em.multi_metrics_evaluation(training_set_path,test_set_path,algo2)
print (" dict 1 :")
print(dict1)
print("dict 2 :")
print(dict2)

print("------ End Of Case 2 \n\n")

# Case 3 : 6 month
print("-------Case 3: 6month")

training_set_path="/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/6Months/6months.csv"
file_path_training_data = os.path.expanduser(training_set_path)
reader = Reader(line_format='user item rating timestamp', sep=',')
training_data = Dataset.load_from_file(file_path_training_data, reader=reader)

sim_options1 = {'name': 'pearson_baseline', 'user_based': False}
sim_options2 = {'name': 'pearson_baseline', 'user_based': True}
algo1 = KNNWithMeans(sim_options=sim_options1)
algo2 = KNNWithMeans(sim_options=sim_options2)
dict1 = Em.multi_metrics_evaluation(training_set_path,test_set_path,algo1)
dict2 = Em.multi_metrics_evaluation(training_set_path,test_set_path,algo2)
print (" dict 1 :")
print(dict1)
print("dict 2 :")
print(dict2)

print("------ End Of Case 3 \n\n")

# Case 4 : 12 month
print("-------Case 4: 12month")

training_set_path="/home/obisama/Desktop/PFE_PFE/pfe/2008/USE_CASE_DATA/1Year/1year.csv"
file_path_training_data = os.path.expanduser(training_set_path)
reader = Reader(line_format='user item rating timestamp', sep=',')
training_data = Dataset.load_from_file(file_path_training_data, reader=reader)
sim_options1 = {'name': 'pearson_baseline', 'user_based': False}
sim_options2 = {'name': 'pearson_baseline', 'user_based': True}
algo1 = KNNWithMeans(sim_options=sim_options1)
algo2 = KNNWithMeans(sim_options=sim_options2)
dict1 = Em.multi_metrics_evaluation(training_set_path,test_set_path,algo1)
dict2 = Em.multi_metrics_evaluation(training_set_path,test_set_path,algo2)
print (" dict 1 :")
print(dict1)
print("dict 2 :")
print(dict2)

print("------ End Of Case 4 \n\n")

