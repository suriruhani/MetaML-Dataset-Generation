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
import pickle

spect_train = pd.read_csv("Dataset/SPECT.train", sep= ',', header= None)
spect_test = pd.read_csv("Dataset/SPECT.test", sep= ',', header= None)
spect = pd.read_csv("Dataset/SPECT.test", sep= ',', header= None)

X_train = spect_train.values[:, 1:22]
y_train = spect_train.values[:,0]

X_test = spect_test.values[:, 1:22]
y_test = spect_test.values[:,0]

X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# dataset structure:
# id, Y, X1......X22  weight
# 0,  1,  2........23,  24
# () --> for within pass calculations
# weight = 1 at start, doubled for every wrong classification
# fold_i = x where x is number of times the instance is included in ith fold

number_of_pass = 10
fold_per_boost = 10
dataset = np.concatenate((spect_train, spect_test), axis=0)
size = len(dataset)
id = [[x] for x in range(size)]
dataset = np.append(id, dataset, axis=1)

weights = np.ones((size,1), dtype=int) # add weights
dataset = np.append(dataset, weights, axis=1)

# for f in range(num_boost): # add folds
#     dataset = np.append(dataset, np.zeros((size,1), dtype=int), axis=1)

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
past_ids = []
for k in range(number_of_pass): # number of resampling
    # form this pass dataset
    print(f'-----------PASS {k+1}-----------')
    dataset_now = []
    weights_now = np.cumsum(dataset[:, 24]/np.sum(dataset[:, 24]),axis=0)
    for _ in range(size):
        roll = random()
        index = binary_search(weights_now, roll)
        dataset_now.append(dataset[index])
    with open(f"dataset_pass{k+1}.txt", 'w') as f:
        for row in dataset_now:
            f.write('%s\n' % row)
    # make folds and train, test
    fold_size = floor(size/fold_per_boost)
    y_all = list(zip(*dataset_now))[1]
    this_ids = list(zip(*dataset_now))[0]
    common_0 = [x[0] for x in dataset_now if x[1] == 0 and x[0] in past_ids]
    common_1 = [x[0] for x in dataset_now if x[1] == 1 and x[0] in past_ids]
    print("Class 0: ", y_all.count(0), "Class 1: ", y_all.count(1))
    # print("Common values: ", [x for x in this_ids if x in past_ids])
    for f in range(fold_per_boost):
        print(f'Fold {f+1}')
        X_test = dataset_now[fold_size*f:fold_size*(f+1)][2:24]
        y_test = list(zip(*X_test))[1]
        id_test = list(zip(*X_test))[0]
        if (f == 0):
            X_train = dataset_now[fold_size*(f+1):][2:24]
        elif (f == fold_per_boost-1):
            X_train = dataset_now[:fold_size*(f-1)][2:24]
        else:
            X_train = np.concatenate((dataset_now[:fold_size*(f)][2:24], dataset_now[fold_size*(f+1):][2:24]), axis=0)
        y_train = list(zip(*X_train))[1]
        model = tree.DecisionTreeClassifier()
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        # prediction = 1 for class 1, 0 for class 0, -1 for not in this fold
        # score = 1 for correct classification, 0 for misclassification, -1 for not in this fold
        total = len(id_test)
        miss = 0
        common_id = 0
        common_miss = 0
        common_0_miss = 0
        common_1_miss = 0
        for i, id in enumerate(id_test):
            if id in past_ids:
                common_id += 1
            pred = prediction[i] # store prediction for this run
            # check for score by matching y label to prediction
            score = 1 if (pred == dataset[id][1]) else 0
            if (score == 0):
                miss += 1
                if id in past_ids:
                    common_miss += 1
                if id in common_0:
                    common_0_miss += 1
                if id in common_1:
                    common_1_miss += 1
                dataset[id][24] *= 2 # double weight if misclassified
        print("accuracy:", accuracy_score(y_test, prediction)*100, "%",
              "missed:", miss*100/total, "%")
        if common_id != 0:
            print("accuracy in common values:", (1 - (common_miss/common_id))*100, "%")
        if len(common_0) != 0:
            print("accuracy in common Class 0 values:", (1 - (common_0_miss/len(common_0)))*100, "%")
        if len(common_1) != 0:
            print("accuracy in common Class 1 values:", (1 - (common_1_miss/len(common_1)))*100, "%")
    past_ids = this_ids

