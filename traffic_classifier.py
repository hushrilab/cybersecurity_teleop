import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
import csv
import numpy as np
import random
from random import shuffle
import math
import time
import warnings
import pandas as pd


import sklearn
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
#Classifiers
from xgboost import XGBClassifier
import xgboost as xgb

#Eval Metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from scipy import interp

from termcolor import colored 
from itertools import combinations

sklearn.set_config(assume_finite=True)

def PrintColored(string, color):
    print(colored(string, color))


def mergeDatasets(dir_path):
    all_dataframes = []
    dir_names = os.listdir(dir_path)  # list all files in the directory
    dir_names = [dn for dn in dir_names if dn.startswith('capPressKey')]  # keep only .csv files

    # iterate over each file and load it into a pandas Series, then append to list
    for dn in dir_names:
        fn = os.path.join(dir_path, dn) + "/Stats" + "/Stats.csv"
        df = pd.read_csv(fn, header=None)  # transpose the DataFrame so we have columns and one row
        df.columns = df.iloc[0]  # make the first row as header
        df = df.drop(df.index[0])  # drop the first row
        df = df.loc[:]  # keep all columns
#         df = df.apply(pd.to_numeric)  # convert values to numeric
        all_dataframes.append(df)

    # concatenate all dataframes
    combined_df = pd.concat(all_dataframes)

    # save the combined dataframe to a new csv file
    combined_df.to_csv('combined.csv', index=False)

    
def preprocess(file_path):
    #Load Datasets
    f = open(file_path, 'r')
    reader = csv.reader(f, delimiter=',')
    pre_data = list(reader)

    #Convert data to floats (and labels to integers)
    features_id = pre_data[0]
    data = []
    for i in pre_data[1:]:
        int_array = []
        for pl in i[:-1]:
            int_array.append(float(pl))
        int_array.append(0)
        data.append(int_array)
        
    #Shuffle both datasets
    shuffled_data = random.sample(data, len(data))

    #Build label tensors
    labels = []
    for i in shuffled_data:
        labels.append(int(i[len(data[0])-1]))

    #Take label out of data tensors
    for i in range(0, len(shuffled_data)):
        shuffled_data[i].pop()

    #Create training sets by combining the randomly selected samples from each class
    train_x = shuffled_data
    train_y = labels

    #Shuffle positive/negative samples for CV purposes
    x_shuf = []
    y_shuf = []
    index_shuf = list(range(len(train_x)))
    shuffle(index_shuf)
    for i in index_shuf:
        x_shuf.append(train_x[i])
        y_shuf.append(train_y[i])

    return x_shuf, y_shuf, features_id

def runClassificationKFold_CV(data_path):
    #Set fixed randomness
    np.random.seed(1)
    random.seed(1)

    dataset_fraction = 1.0
    train_x, train_y, features_id = preprocess(data_path)
    model = XGBClassifier()

    #Report Cross-Validation Accuracy
    cv = StratifiedKFold(n_splits=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    train_times = []
    test_times = []
    importances = []

    #Split the data in k-folds, perform classification, and report ROC
    i = 0
    for train, test in cv.split(train_x, train_y):

        start_train = time.time()
        model = model.fit(np.asarray(train_x)[train], np.asarray(train_y)[train])
        end_train = time.time()
        train_times.append(end_train - start_train)

        start_test = time.time()
        probas_ = model.predict_proba(np.asarray(train_x)[test])
        end_test = time.time()
        test_times.append(end_test - start_test)

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(np.asarray(train_y)[test], probas_[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)


        if(roc_auc < 0.5):
            roc_auc = 1 - roc_auc
            fpr = [1 - e for e in fpr]
            fpr.sort()
            tpr = [1 - e for e in tpr]
            tpr.sort()

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
#     print "Model AUC: " + "{0:.3f}".format(mean_auc)

    if(mean_auc < 0.5):
        mean_auc = 1 - mean_auc
#         print "Inverting ROC curve - new auc: " + str(mean_auc)
        fpr = [1 - e for e in fpr]
        fpr.sort()
        tpr = [1 - e for e in tpr]
        tpr.sort()


#     print "10-Fold AUC: " + "{0:.3f}".format(mean_auc)


    #Figure properties
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    #Compute Standard Deviation between folds
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'$\pm$ ROC Std. Dev.')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2, color='orange', label = 'Random Guess')
    ax1.grid(color='black', linestyle='dotted')

    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate', fontsize='x-large')
    plt.ylabel('True Positive Rate', fontsize='x-large')
    plt.legend(loc='lower right', fontsize='large')

    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)
    plt.show()

    return mean_tpr, mean_fpr, mean_auc