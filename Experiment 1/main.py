import numpy as np
import pandas as pd
from random import random
from sklearn.metrics import accuracy_score
from sklearn import tree
from matplotlib import style
import _thread
import threading
import time
style.use('fivethirtyeight')

def binary_search(arr, comp):
    if comp < arr[0]:
        return 0
    low = 0
    high = len(arr) - 1
    mid = ((high + low)//2)
    while not (comp > arr[mid-1] and comp <= arr[mid]):
        if comp <= arr[mid-1]:
            high = mid - 1
            if high <= 0:
                return 0
        else:
            low = mid + 1
            if low >= len(arr) - 1:
                return len(arr) - 1
        mid = ((high + low)//2)
    return mid

def flatten_2d_list(ini_list):
    return [j for sub in ini_list for j in sub]

def main(path, sep, is_last):

    # id, Y, X(1)......X(num_attr),  weight
    # 0,  1,  2........num_attr+1,  num_attr+2

    dataset = pd.read_csv(path, sep=sep, header=None)
    num_attr = len(dataset.columns) - 1

    if is_last:
        dataset = dataset.reindex([num_attr] + [x for x in range(num_attr)], axis=1)
        dataset.columns = range(num_attr+1)

    filename = path.split("/")[-1]
    file = open("Results/Policy_3/"+filename, 'w')

    number_of_pass = 10
    fold_per_boost = 10
    size = len(dataset)
    id = [[x] for x in range(size)]
    dataset = np.append(id, dataset, axis=1)

    weights = np.ones((size,1), dtype=int) # add weights
    dataset = np.append(dataset, weights, axis=1)

    y_values = [] # for r squared calculation

    for k in range(number_of_pass): # number of resampling
        # form this pass dataset
        file.write(f'-----------PASS {k+1}-----------\n')
        if (k != 0): # resample from second pass onwards
            dataset_now = []

            class1 = []
            class0 = []
            for r in dataset:
                if r[1] == 1:
                    class1.append(r)
                elif r[1] == 0:
                    class0.append(r)

            size_of_0 = len(dataset)//2
            size_of_1 = len(dataset) - size_of_0

            weights_of_0 = np.cumsum(list(zip(*class0))[num_attr+2]/np.sum(list(zip(*class0))[num_attr+2]), axis=0)
            weights_of_1 = np.cumsum(list(zip(*class1))[num_attr+2]/np.sum(list(zip(*class1))[num_attr+2]), axis=0)

            for _ in range(size_of_0):
                roll = random()
                index = binary_search(weights_of_0, roll)
                dataset_now.append(class0[index])

            for _ in range(size_of_1):
                roll = random()
                index = binary_search(weights_of_1, roll)
                dataset_now.append(class1[index])

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
        file.write("class length 0 & 1:" + str(len(class0)) + " " + str(len(class1)) + "\n")
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

        accuracy_sum = 0

        for f in range(fold_per_boost):
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
            miss = 0
            wrong_zero = wrong_one = zero = one = 0
            wrong_zero_id = []
            wrong_one_id = []
            for i, id in enumerate(id_test):
                pred = prediction[i] # store prediction for this run
                # check for score by matching y label to prediction
                score = 1 if (pred == dataset[int(id)][1]) else 0
                is_zero = True if dataset[int(id)][1] == 0 else False
                if is_zero:
                    zero += 1
                else:
                    one += 1

                if (score == 0):
                    miss += 1
                    if is_zero:
                        wrong_zero += 1
                        wrong_zero_id.append(id)
                    else:
                        wrong_one += 1
                        wrong_one_id.append(id)

            for id in wrong_zero_id:
                dataset[int(id)][num_attr+2] *= (1+wrong_zero/zero)

            for id in wrong_one_id:
                dataset[int(id)][num_attr+2] *= (1+wrong_one/one)

            accuracy_value = accuracy_score(y_test, prediction)*100
            accuracy_sum += accuracy_value
            file.write(str(accuracy_value) + " %\n")

        y_values.append(accuracy_sum/fold_per_boost)

    file.close()

    x_values = range(1,1+number_of_pass)
    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print(r_squared)

for i in range(41):
    main(f"Dataset/ECOC/{i}.txt", ",", True)


