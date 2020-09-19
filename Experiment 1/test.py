import numpy as np
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


balance_data = pd.read_csv("balance-scale.data", sep= ',', header= None)
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
    
def dt_no_cv():

    #X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)



    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                   max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)

    y_pred = clf_gini.predict(X_test)
    print("prediction = ")
    print(y_pred)

    print("accuracy = ", accuracy_score(y_test,y_pred)*100)

    cm = confusion_matrix(y_test, y_pred)
    print("confusion matrix: ")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f'tn={tn}, fp={fp}, fn={fn}, tp={tp}')

def dt_10_cv():
    
    kf10 = StratifiedKFold(n_splits=10, shuffle=False)
    clf = tree.DecisionTreeClassifier(random_state=20)
    #model = clf.fit(X_train, y_train)
    
    accuracy_model = []

    for train_index, test_index in kf10.split(X_all, y_all):
         # print(train_index, test_index)
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        model = clf.fit(X_train, y_train)
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test),
                                             normalize=True)*100)

    print(accuracy_model)

def dt_ada_boost():
    AdaBoost = AdaBoostClassifier(n_estimators=400,learning_rate=1,algorithm='SAMME')

    AdaBoost.fit(X_all,y_all)

    prediction = AdaBoost.score(X_all,y_all)

    print('The accuracy is: ',prediction*100,'%')

def dt_boost():

    number_of_base_learners = 10

    fig = plt.figure(figsize=(10,10))
    ax0 = fig.add_subplot(111)

    model = Boosting(number_of_base_learners)
    model.fit(X_train, y_train)
    model.predict(X_test, y_test)

    ax0.plot(range(len(model.accuracy)),model.accuracy,'-b')
    ax0.set_xlabel('# models used for Boosting ')
    ax0.set_ylabel('accuracy')
    print('With a number of ',number_of_base_learners,'base models we receive an accuracy of ',model.accuracy[-1]*100,'%')    
                     
    plt.show()

def dt_cv_boost():

    number_of_base_learners = 10
    k = 10
    model = Boosting(number_of_base_learners)
    model.get_fold_result(X_all, y_all, k)
    
    
dt_cv_boost()




