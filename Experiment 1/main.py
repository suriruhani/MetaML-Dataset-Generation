import matplotlib.pyplot as plt
import numpy as np
from math import floor
import pandas as pd
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

spect_train = pd.read_csv("SPECT.train", sep= ',', header= None)
spect_test = pd.read_csv("SPECT.test", sep= ',', header= None)
spect = pd.read_csv("SPECT.test", sep= ',', header= None)

X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

X_train = spect_train.values[:, 1:22]
y_train = spect_train.values[:,0]

X_test = spect_test.values[:, 1:22]
y_test = spect_test.values[:,0]

X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# dataset structure:
# X1......X22 , Y, weight, fold_1....fold_10, prediction, score
# 0........21,  22, 23, 24......,33, 34, 35
# weight = 1 at start, doubled for every wrong classification
# fold_i = x where x is number of times the instance is included in ith fold
# prediction = 1 for class 1, 0 for class 0, -1 for not in this fold
# score = 1 for correct classification, 0 for misclassification, -1 for not in this fold

num_boost = 10
dataset = np.concatenate((spect_train, spect_test), axis=0)
fold_size = floor(len(dataset)/num_boost)

weights = np.ones((size,1), dtype=int)
dataset = np.append(dataset, weights, axis=1)

for f in range(num_boost):
    dataset = np.append(dataset, np.zeros((size,1), dtype=int), axis=1)

dataset = np.append(dataset, np.full((size, 1), -1), axis=1)
dataset = np.append(dataset, np.full((size, 1), -1), axis=1)

for k in range(num_boost):
    print(f"-----------PASS {k+1}-----------")
    for i in range(22+2, 22+2+num_boost):
        model = tree.DecisionTreeClassifier()
        dataset[:, 23] = np.cumsum(dataset[:,23]/np.sum(dataset[:,23]),axis=0)
        
        fold = dataset[dataset[:,i] == 1][:,range(24)]
        X_train = []
        for row in fold:
            X_train.extend([row[:22]]*row[23]) # include it as many times as the weight
        y_train = dataset[dataset[:,i] == 1][:,22+2+num_boost]
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_train)
        print(accuracy_score(y_train, prediction)*100, "%") 
        p_count = 0
        for j in range(size):
            if (dataset[j][i] == 1): # in this fold
                dataset[j][34] = prediction[p_count] # store prediction for this run
                p_count += 1 # check for score by matching y label to prediction
                dataset[j][35] = 1 if (dataset[j][34] == dataset[j][22]) else 0
                if (dataset[j][35] == 0):
                    dataset[j][23] *= 2 # double weight if misclassified
        
    
    
def binary_search(arr, comp):
    if comp < arr[0]:
        return 0
    low = 0
    high = len(arr) - 1
    mid = floor((high + low)/2)
    while not (comp > arr[mid-1] and comp <= arr[mid]):
        print(low, high, mid)
        if comp <= arr[mid-1]:
            high = mid - 1
        else:
            low = mid + 1
        mid = floor((high + low)/2)
    return mid











