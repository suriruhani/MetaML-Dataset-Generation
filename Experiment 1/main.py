import numpy as np
import pandas as pd
from random import random, randint
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from math import isnan
from sklearn.naive_bayes import GaussianNB

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

def policy_1a(dataset, id, w_col, factor):
    dataset[id][w_col] *= factor

def policy_1b(dataset, id, w_col):
    dataset[id][w_col] += 1


def main(path, sep, is_last, policy_file, r2_file, acc_inc_file, acc_dec_file, acc_eq_file, rep_file):

    # id, Y, X(1)......X(num_attr),  weight
    # 0,  1,  2........num_attr+1,  num_attr+2

    dataset = pd.read_csv(path, sep=sep, header=None)
    num_attr = len(dataset.columns) - 1

    if is_last:
        dataset = dataset.reindex([num_attr] + [x for x in range(num_attr)], axis=1)
        dataset.columns = range(num_attr+1)

    # print("STATS --->", num_attr, len(dataset))

    filename = path.split("/")[-1]
    file = open("Results/"+filename, 'w')

    number_of_pass = 10
    fold_per_boost = 10
    size = len(dataset)
    id = [[x] for x in range(size)]
    dataset = np.append(id, dataset, axis=1)

    weights = np.ones((size,1), dtype=int) # add weights
    dataset = np.append(dataset, weights, axis=1)

    y_values = []
    # for r squared calculation

    total_gradient = number_of_pass - 1
    prev_acc = -1
    rise = fall = 0

    ratio_by_class = [0,0]

    for r in dataset:
        if r[1] == 1:
            ratio_by_class[1] += 1
        elif r[1] == 0:
            ratio_by_class[0] += 1
    ratio_by_class[1] /= size
    ratio_by_class[0] /= size

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


            weights_total = np.cumsum(list(zip(*dataset))[num_attr+2]/np.sum(list(zip(*dataset))[num_attr+2]), axis=0)

            weights_of_0 = np.cumsum(list(zip(*class0))[num_attr+2]/np.sum(list(zip(*class0))[num_attr+2]), axis=0)
            weights_of_1 = np.cumsum(list(zip(*class1))[num_attr+2]/np.sum(list(zip(*class1))[num_attr+2]), axis=0)

            current_ratio = [0,0]
            class_allow = [True, True]
            while len(dataset_now) < size:
                roll = random()
                index = binary_search(weights_total, roll)
                class_of_this = dataset[index][1]
                current_class_ratio = current_ratio[int(class_of_this)]/len(dataset_now) if current_ratio[int(class_of_this)]!= 0 else 0
                idea_class_ratio = 0.5
                # ratio of a class lower bounded by original ratio and upper bounded by 50%
                if (current_class_ratio <= idea_class_ratio):
                    dataset_now.append(dataset[index])
                    current_ratio[int(class_of_this)] += 1
                else:
                    class_allow[int(class_of_this)] = False
                    break

            while len(dataset_now) < size:
                roll = random()
                index = binary_search(weights_of_0, roll) if class_allow[0] else binary_search(weights_of_1, roll)
                dataset_now.append(class0[index]) if class_allow[0] else dataset_now.append(class1[index])

        else: # use entire dataset for first pass
            dataset_now = dataset

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

            # knn helper
            # hepler_model = KNeighborsClassifier(n_neighbors=1)
            hepler_model = GaussianNB()
            hepler_model.fit(X_train, y_train)
            helper_prediction = hepler_model.predict(X_test)

            # prediction = 1 for class 1, 0 for class 0, -1 for not in this fold
            # score = 1 for correct classification, 0 for misclassification, -1 for not in this fold

            all_ids = list(zip(*dataset_now.copy()))[0]

            for i, id in enumerate(id_test):
                pred = prediction[i] # store prediction for this run
                # check for score by matching y label to prediction
                score = 1 if (pred == dataset[int(id)][1]) else 0

                if (score == 0):
                    policy_1a(dataset, int(id), num_attr+2, 2)

            accuracy_value = accuracy_score(y_test, prediction)*100
            accuracy_sum += accuracy_value
            file.write(str(accuracy_value) + " %\n")

        iteration_accuracy = accuracy_sum/fold_per_boost
        y_values.append(iteration_accuracy)
        file.write(f"-----{iteration_accuracy}%-----\n")
        if prev_acc != -1:
            tolerance = 0.05 #5%
            rise += 1 if ((iteration_accuracy) > (1+tolerance)*(prev_acc)) else 0
            fall += 1 if ((iteration_accuracy) < (1-tolerance)*(prev_acc)) else 0

        prev_acc = iteration_accuracy

    x_values = range(1,1+number_of_pass)
    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print(r_squared)

    unique_test = list(zip(*dataset_now.copy()))[0]
    dataset_set = len(np.unique(unique_test))
    rep_percentage = (size - dataset_set)/size

    file.write("Overall dataset R square: " + str(r_squared) + "\n")
    file.write("Accuracy increase ratio: " + str(rise/total_gradient) + "\n")
    file.write("Accuracy decrease ratio: " + str(fall/total_gradient) + "\n")
    file.write("Accuracy equal ratio: " + str((total_gradient-fall-rise)/total_gradient) + "\n")
    file.write("Repeatition percentage: " + str(rep_percentage) + "\n")
    file.close()

    r2_file.write(str(r_squared) + "\n")
    acc_inc_file.write(str(rise/total_gradient) + "\n")
    acc_dec_file.write(str(fall/total_gradient) + "\n")
    acc_eq_file.write(str((total_gradient-fall-rise)/total_gradient) + "\n")
    rep_file.write(str(rep_percentage) + "\n")

    policy_file.write(filename + " : R^2 = " + str(r_squared) + " Rise ratio = "
                      + str(rise/total_gradient) + " Unique % " + str(rep_percentage) + "\n")

    return r_squared

chosen_datasets = []
# datasets with less than 8 classes originally
valid_datasets = (list(range(1,155)) + list(range(20630, 21374)))
                  # + list(range(23420, 26882)) + list(range(27392, 27510))
                # + list(range(28533, 28658)))

# choose 100 randomly
test_dataset_count = 10
while len(chosen_datasets) < test_dataset_count and len(valid_datasets) > 0:
    val = randint(0,len(valid_datasets)-1)
    try:
        path = "/Users/suriruhani/OneDrive - National University of Singapore/FYP/Meta-learning/Datasets/UCI/chosen/"+str(valid_datasets[val])+".txt"
        # path = "/Volumes/My Passport/[-] Storage/Meta-learning/Datasets/UCI/chosen/"+str(i)+".txt"
        dataset = pd.read_csv(path, sep=",", header=None)
    except:
        print("fail")
        valid_datasets.pop(val)
        pass
    else:
        if (len(dataset.columns) - 1 <= 30 and len(dataset) <= 3000):
            chosen_datasets.append(valid_datasets.pop(val))
            print("added " + str(len(chosen_datasets)))
        else:
            valid_datasets.pop(val)
            print("pass")

policy_sum = 0
valid = 0
overall = open("Results/3_Overall.txt", 'a')
r2_file = open("Results/R2_Overall.txt", 'a')
acc_inc_file = open("Results/Acc_Increase_Overall.txt", 'a')
acc_dec_file = open("Results/Acc_Decrease_Overall.txt", 'a')
acc_eq_file = open("Results/Acc_Equal_Overall.txt", 'a')
rep_file = open("Results/Rep_Overall.txt", 'a')
overall.truncate(0)
r2_file.truncate(0)
acc_inc_file.truncate(0)
acc_dec_file.truncate(0)
acc_eq_file.truncate(0)
rep_file.truncate(0)

for i in chosen_datasets:
    print("NOW TRYING --->", i)
    dataset_result = main(f"/Users/suriruhani/OneDrive - National University of Singapore/FYP/Meta-learning/Datasets/UCI/chosen/{i}.txt",
                          ",", True, overall, r2_file, acc_inc_file, acc_dec_file, acc_eq_file, rep_file)
    # main(f"/Volumes/My Passport/[-] Storage/Meta-learning/Datasets/UCI/chosen/{i}.txt",
    #                       ",", True, overall)

    if not isnan(dataset_result):
        policy_sum += dataset_result
        valid+=1

overall.close()
r2_file.close()
acc_inc_file.close()
acc_dec_file.close()
acc_dec_file.close()
rep_file.close()
print("OVERALL ---->", policy_sum/valid)

