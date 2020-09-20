import matplotlib.pyplot as plt
import numpy as np
from math import floor
import pandas as pd
from random import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from boosting import Boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.model_selection import cross_validate
import scipy.stats as sps

dataset = pd.read_csv("Dataset/balance-scale.data", sep= ',', header= None)

X = dataset.values[:, 1:5]
y = dataset.values[:, 0]

# dataset structure:
# Y, X1, X2, X3, X4  weight, fold_1....fold_10
# 0,  1, 2, 3,   4,    5,     6, ..... 15
# weight = 1 at start, doubled for every wrong classification
# fold_i = x where x is number of times the instance is included in ith fold

num_boost = 10
size = len(dataset)
fold_size = floor(size/num_boost)

weights = np.ones((size,1), dtype=int) # add weights
dataset = np.append(dataset, weights, axis=1)

for f in range(num_boost): # add folds
    dataset = np.append(dataset, np.zeros((size,1), dtype=int), axis=1)

def binary_search(arr, comp):
    if comp < arr[0]:
        return 0
    low = 0
    high = len(arr) - 1
    mid = floor((high + low)/2)
    while not (comp > arr[mid-1] and comp <= arr[mid]):
        if comp <= arr[mid-1]:
            high = mid - 1
        else:
            low = mid + 1
        mid = floor((high + low)/2)
    return mid

for k in range(num_boost): # number of reweighing
    print(f'-----------PASS {k+1}-----------')
    for i in range(6, 15): # number of resampling
        weights_now = np.cumsum(dataset[:, 5]/np.sum(dataset[:, 5]),axis=0)
        for _ in range(fold_size):
            roll = random()
            index = binary_search(weights_now, roll)
            dataset[:, i][index] += 1

    for i in range(6, 15): # number of resampling
        fold = dataset[dataset[:,i] > 0][:,range(6)]
        X_train = []
        y_train = []
        for row in fold:
            X_train.extend([row[1:5]]*row[5])
            # include it as many times as the weight
            y_train.extend([row[0]]*row[5])
        model = tree.DecisionTreeClassifier()
        model = model.fit(X_train, y_train)
        # test on full dataset
        X_test = dataset[:, 1:5]
        y_test = dataset[:, 0]
        prediction = model.predict(X_test)
        print(accuracy_score(y_test, prediction)*100, "%")
        # prediction = 1 for class 1, 0 for class 0, -1 for not in this fold
        # score = 1 for correct classification, 0 for misclassification, -1 for not in this fold
        for j in range(size):
            pred = prediction[j] # store prediction for this run
            # check for score by matching y label to prediction
            score = 1 if (pred == dataset[j][0]) else 0
            if (score == 0):
                dataset[j][5] *= 2 # double weight if misclassified
