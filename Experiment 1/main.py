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

def flatten_2d_list(ini_list):
    return [j for sub in ini_list for j in sub]

def main(path, sep, num_attr):

    dataset = pd.read_csv(path, sep=sep, header=None)
    # del dataset[0]
    # print(dataset)
    # temp = dataset.copy()[10]
    # dataset[10] = dataset[1]
    # dataset[1] = temp
    # print(dataset)
    # id, Y, X(1)......X(num_attr),  weight
    # 0,  1,  2........num_attr+1,  num_attr+2

    number_of_pass = 10
    fold_per_boost = 10
    size = len(dataset)
    id = [[x] for x in range(size)]
    dataset = np.append(id, dataset, axis=1)
    # dataset[:,[1,23]] = dataset[:,[23,1]]
    # print(dataset)
    # return

    weights = np.ones((size,1), dtype=int) # add weights
    dataset = np.append(dataset, weights, axis=1)

    for k in range(number_of_pass): # number of resampling
        # form this pass dataset
        print(f'-----------PASS {k+1}-----------')
        if (k != 0): # resample from second pass onwards
            dataset_now = []
            weights_now = np.cumsum(dataset[:, num_attr+2]/np.sum(dataset[:, num_attr+2]), axis=0)
            # print(weights_now)
            for _ in range(size):
                roll = random()
                index = binary_search(weights_now, roll)
                dataset_now.append(dataset[index])
        else: # use entire dataset for first pass
            dataset_now = dataset

        # uncomment for printing each dataset
        # with open(f"dataset_pass{k+1}.txt", 'w') as f:
        #     for row in dataset_now:
        #         f.write('%s\n' % row)
        # make folds and train, test

        class1 = []
        class0 = []
        for r in dataset_now:
            if r[1] == 1:
                class1.append(r)
            elif r[1] == 0:
                class0.append(r)
        print("class length 0 & 1:", len(class0), len(class1))

        strat_folds = [[] for _ in range(fold_per_boost)]

        count_fold = 0
        while len(class0) != 0:
            strat_folds[count_fold].append(class0.pop())
            count_fold += 1
            if count_fold == fold_per_boost:
                count_fold = 0

        count_fold = 0
        while len(class1) != 0:
            strat_folds[count_fold].append(class1.pop())
            count_fold += 1
            if count_fold == fold_per_boost:
                count_fold = 0

        for f in range(fold_per_boost):
            # print(f'Fold {f+1}')
            test = strat_folds[f]
            X_test = [row[2:num_attr+2] for row in test]
            y_test = list(zip(*test))[1]
            id_test = list(zip(*test))[0]

            if (f == 0):
                train = flatten_2d_list(strat_folds[1:])
            elif (f == fold_per_boost-1):
                train = flatten_2d_list(strat_folds[:f-1])
            else:
                train = np.concatenate((flatten_2d_list(strat_folds[:f]), flatten_2d_list(strat_folds[f+1:])), axis=0)

            X_train = [row[2:num_attr+2] for row in train]
            y_train = list(zip(*train))[1]

            model = tree.DecisionTreeClassifier()
            model = model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            # prediction = 1 for class 1, 0 for class 0, -1 for not in this fold
            # score = 1 for correct classification, 0 for misclassification, -1 for not in this fold
            total = len(id_test)
            miss = 0
            for i, id in enumerate(id_test):
                pred = prediction[i] # store prediction for this run
                # check for score by matching y label to prediction
                score = 1 if (pred == dataset[int(id)][1]) else 0
                if (score == 0):
                    miss += 1
                    dataset[int(id)][num_attr+2] *= 2
                    # double weight if misclassified
            print(accuracy_score(y_test, prediction)*100, "%")

main("Dataset/SPECT.test", ",", 22)

