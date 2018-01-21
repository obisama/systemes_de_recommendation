"""
this code is an example on how to calculate euclidian distance between two nodes.
"""

import random
import math
from surprise import SVD
from surprise import Dataset
import unicodecsv
from surprise import evaluate, print_perf
from surprise import KNNWithZScore
from surprise import KNNBasic
import csv
import time
import os
import pandas as pd
import numpy as np
from surprise import Reader
import csv

file = os.path.expanduser('/home/ok/ok/test.csv')
reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)
datatest=Dataset.load_from_file(file,reader=reader)

def shuffle(i_data):
	""" """
 random.shuffle(i_data)
 train_data = i_data[:(len(i_data))//9]
 test_data = i_data[len(i_data)//9:]
 return train_data, test_data
def getdata(filename):
    with open(filename,'rb') as f:
        reader = unicodecsv.reader(f)
        return list(reader)

#data=getdata('/home/ok/ok/test.csv')
data = Dataset.load_from_file('/home/ok/ok/test.csv', Reader(line_format='user item rating timestamp',sep=',',skip_lines=0))

test,train=shuffle(data)
print (len(data))
print (len(test))
print (len(train))
print (len(test)+len(train))
def euclideanDist(x, xi):
 d = []
 for i in x:
     k=0
     d += pow((i[2]-xi[2]),2)
     d[k]= math.sqrt(d)
     k=k+1

 return d
x={(1,'a',1),(1,'b',5)}
x1=(2,'b',4)
print (euclideanDist(x,x1))