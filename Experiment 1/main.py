import numpy as np
from math import floor
import pandas as pd
from random import random
from sklearn.metrics import accuracy_score
from sklearn import tree
from matplotlib import style
style.use('fivethirtyeight')

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

def main(path, sep, num_attr):

    dataset = pd.read_csv(path, sep=sep, header=None)

    # id, Y, X(1)......X(num_attr),  weight
    # 0,  1,  2........num_attr+1,  num_attr+2

    number_of_pass = 10
    fold_per_boost = 10
    size = len(dataset)
    id = [[x] for x in range(size)]
    dataset = np.append(id, dataset, axis=1)

    weights = np.ones((size,1), dtype=int) # add weights
    dataset = np.append(dataset, weights, axis=1)
    # past_ids = []

    for k in range(number_of_pass): # number of resampling
        # form this pass dataset
        print(f'-----------PASS {k+1}-----------')
        dataset_now = []
        weights_now = np.cumsum(dataset[:, num_attr+2]/np.sum(dataset[:, num_attr+2]), axis=0)
        # print(weights_now)
        for _ in range(size):
            roll = random()
            index = binary_search(weights_now, roll)
            dataset_now.append(dataset[index])
        with open(f"dataset_pass{k+1}.txt", 'w') as f:
            for row in dataset_now:
                f.write('%s\n' % row)
        # make folds and train, test
        fold_size = floor(size/fold_per_boost)
        # y_all = list(zip(*dataset_now))[1]
        # this_ids = list(zip(*dataset_now))[0]
        # common_0 = [x[0] for x in dataset_now if x[1] == 0 and x[0] in past_ids]
        # common_1 = [x[0] for x in dataset_now if x[1] == 1 and x[0] in past_ids]
        # print("Class 0: ", y_all.count(0), "Class 1: ", y_all.count(1))
        # print("Common values: ", [x for x in this_ids if x in past_ids])
        for f in range(fold_per_boost):
            # print(f'Fold {f+1}')
            test = dataset_now[fold_size*f:fold_size*(f+1)]
            X_test = [row[2:num_attr+2] for row in test]
            y_test = list(zip(*test))[1]
            id_test = list(zip(*test))[0]
            # print(X_test[0])
            # print(y_test[0])
            # print(id_test[0])
            if (f == 0):
                train = dataset_now[fold_size*(f+1):]
            elif (f == fold_per_boost-1):
                train = dataset_now[:fold_size*(f-1)]
            else:
                train = np.concatenate((dataset_now[:fold_size*(f)], dataset_now[fold_size*(f+1):]), axis=0)
            # print(train)
            X_train = [row[2:num_attr+2] for row in train]
            y_train = list(zip(*train))[1]
            # print(X_train[0])
            # print(y_train[0])
            model = tree.DecisionTreeClassifier()
            model = model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            # prediction = 1 for class 1, 0 for class 0, -1 for not in this fold
            # score = 1 for correct classification, 0 for misclassification, -1 for not in this fold
            total = len(id_test)
            miss = 0
            # common_id = 0
            # common_miss = 0
            common_0_miss = 0
            common_1_miss = 0
            for i, id in enumerate(id_test):
                # print(i, id)
                # if id in past_ids:
                #     common_id += 1
                pred = prediction[i] # store prediction for this run
                # check for score by matching y label to prediction
                # print(pred)
                score = 1 if (pred == dataset[id][1]) else 0
                if (score == 0):
                    miss += 1
                    # if id in past_ids:
                    #     common_miss += 1
                    # if id in common_0:
                    #     common_0_miss += 1
                    # if id in common_1:
                    #     common_1_miss += 1
                    dataset[id][num_attr+2] *= 2
                    # double weight if misclassified
            print(accuracy_score(y_test, prediction)*100, "%")
                  # "missed:", miss*100/total, "%")
            # if common_id != 0:
            #     print("accuracy in common values:", (1 - (common_miss/common_id))*100, "%")
            # if len(common_0) != 0:
            #     print("accuracy in common Class 0 values:", (1 - (common_0_miss/len(common_0)))*100, "%")
            # if len(common_1) != 0:
            #     print("accuracy in common Class 1 values:", (1 - (common_1_miss/len(common_1)))*100, "%")
        # past_ids = this_ids

main("Dataset/balance-scale.data", ",", 4)

