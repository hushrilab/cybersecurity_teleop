import matplotlib
# matplotlib.use('Agg')
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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from scipy import interp

from termcolor import colored 
from itertools import combinations

sklearn.set_config(assume_finite=True)

def PrintColored(string, color):
    print(colored(string, color))
    
def mergeDatasets(dir_path):
    le = preprocessing.LabelEncoder()
    action_names = os.listdir(dir_path)  # list all dirs in the directory
    all_classes_dataframes = []  # list to hold combined dataframes of all classes

    # iterate over each directory
    for an in action_names:
        if ("." in an):
            continue
        dfs_dict = {}  # Initialize an empty dictionary to hold dataframes indexed by filename
        features = os.listdir(os.path.join(dir_path, an))  # list all features in the directory
        for feature in features:
            if ("." in feature):
                continue
            fns = os.listdir(os.path.join(dir_path, an, feature))  # list all files in the directory
            for fn in fns:
                if (".DS" in fn) or (".ipynb_checkpoints" in fn):
                    continue
                file_path = os.path.join(dir_path, an, feature, fn)
                if os.path.getsize(file_path) == 0:
                    print(f"Skipping empty file: {file_path}")
                    continue
                print("Attempting to read:", file_path)
                df = pd.read_csv(file_path, header=None)
                df.columns = df.iloc[0]
                df = df.drop(df.index[0])

                # Check if filename already exists in the dictionary
                if fn in dfs_dict:
                    if 'Class' in df.columns:
                        df = df.drop(columns='Class')  # drop 'class' column
                
                    # If it does, concatenate the new dataframe with the existing one
                    dfs_dict[fn] = pd.concat([dfs_dict[fn], df], axis=1)
                else:
                    # If it doesn't, add the new dataframe to the dictionary
                    dfs_dict[fn] = df
                df['Class'] = an
                # get a list of columns
        # Convert the dictionary to a list of dataframes
        all_dataframes = list(dfs_dict.values())

        # concatenate all dataframes
        combined_df = pd.concat(all_dataframes)

        # Add the combined dataframe to the list of all classes' dataframes
        all_classes_dataframes.append(combined_df)

    # Concatenate all classes dataframes into one
    total_df = pd.concat(all_classes_dataframes)
    total_df = total_df.drop(total_df.columns[-1], axis=1)
    cols = list(total_df.columns)
    # move the column to end of list using index, pop and append
    cols.append(cols.pop(cols.index('Class')))
    # use this reordered list to reorder the dataframe
    total_df = total_df.loc[:, cols]
    print(total_df.columns)
    total_df['Class'] = le.fit_transform(total_df['Class'])
    # save the combined dataframe to a new csv file
    total_df.to_csv(os.path.join(dir_path, 'all_classes.csv'), index=False)

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
        int_array.append(int(i[-1]))
        data.append(int_array)
        
    #Shuffle datasets
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
    print(len(x_shuf))
    print(x_shuf[1])
    print(len(y_shuf))
#     print(features_id)

    return x_shuf, y_shuf, features_id

# def runClassificationKFold_CV(data_path):
#     #Set fixed randomness
#     np.random.seed(1)
#     random.seed(1)

#     dataset_fraction = 1.0
#     train_x, train_y, features_id = preprocess(data_path)
#     model = XGBClassifier()

#     #Report Cross-Validation Accuracy
#     cv = StratifiedKFold(n_splits=10)
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     train_times = []
#     test_times = []
#     importances = []

#     #Split the data in k-folds, perform classification, and report ROC
#     i = 0
#     for train, test in cv.split(train_x, train_y):

#         start_train = time.time()
#         model = model.fit(np.asarray(train_x)[train], np.asarray(train_y)[train])
#         end_train = time.time()
#         train_times.append(end_train - start_train)

#         start_test = time.time()
#         probas_ = model.predict_proba(np.asarray(train_x)[test])
#         end_test = time.time()
#         test_times.append(end_test - start_test)

#         # Compute ROC curve and area under the curve
#         fpr, tpr, thresholds = roc_curve(np.asarray(train_y)[test], probas_[:, 1], pos_label=1)
#         roc_auc = auc(fpr, tpr)


#         if(roc_auc < 0.5):
#             roc_auc = 1 - roc_auc
#             fpr = [1 - e for e in fpr]
#             fpr.sort()
#             tpr = [1 - e for e in tpr]
#             tpr.sort()

#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         aucs.append(roc_auc)
#         i += 1

#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)


#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     print ("Model AUC: " + "{0:.3f}".format(mean_auc))

#     if(mean_auc < 0.5):
#         mean_auc = 1 - mean_auc
# #         print "Inverting ROC curve - new auc: " + str(mean_auc)
#         fpr = [1 - e for e in fpr]
#         fpr.sort()
#         tpr = [1 - e for e in tpr]
#         tpr.sort()


#     print ("10-Fold AUC: " + "{0:.3f}".format(mean_auc))


#     #Figure properties
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)

#     std_auc = np.std(aucs)

#     plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2, alpha=.8)

#     #Compute Standard Deviation between folds
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3, label=r'$\pm$ ROC Std. Dev.')

#     ax1.plot([0, 1], [0, 1], 'k--', lw=2, color='orange', label = 'Random Guess')
#     ax1.grid(color='black', linestyle='dotted')

#     plt.title('Receiver Operating Characteristic (ROC)')
#     plt.xlabel('False Positive Rate', fontsize='x-large')
#     plt.ylabel('True Positive Rate', fontsize='x-large')
#     plt.legend(loc='lower right', fontsize='large')

#     plt.setp(ax1.get_xticklabels(), fontsize=14)
#     plt.setp(ax1.get_yticklabels(), fontsize=14)
#     plt.show()

#     return mean_tpr, mean_fpr, mean_auc

from sklearn.metrics import precision_score, recall_score, f1_score

def runClassificationKFold_CV(data_path, feature_names):
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

    # Initialize lists to store the metrics
    precisions = []
    recalls = []
    f1_scores = []

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

        # Compute precision, recall, and F1 score
        y_pred = model.predict(np.asarray(train_x)[test])
        y_true = np.asarray(train_y)[test]

        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        cm = confusion_matrix(y_true, y_pred)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        print("Confusion Matrix: ")
        print(cm)
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print ("Model AUC: " + "{0:.3f}".format(mean_auc))

    if(mean_auc < 0.5):
        mean_auc = 1 - mean_auc
#         print "Inverting ROC curve - new auc: " + str(mean_auc)
        fpr = [1 - e for e in fpr]
        fpr.sort()
        tpr = [1 - e for e in tpr]
        tpr.sort()


    print ("10-Fold AUC: " + "{0:.3f}".format(mean_auc))
    
    # Compute the mean values of precision, recall, and F1 score
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1_score = np.mean(f1_scores)

    print("Mean Precision: {:.3f}".format(mean_precision))
    print("Mean Recall: {:.3f}".format(mean_recall))
    print("Mean F1 Score: {:.3f}".format(mean_f1_score))
    feature_importance = model.feature_importances_
    mapper = {'f{0}'.format(i): v for i, v in enumerate(feature_names)}

    # Sorted by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    for index in sorted_idx:
        print([mapper[f'f{index}'], feature_importance[index]])

    # Plot the importances with actual feature names
    ax = xgb.plot_importance(model, importance_type='weight', title='Feature importances', xlabel='Weight')
    xgb_labels = ax.get_yticklabels()
    ax.set_yticklabels([mapper[label.get_text()] for label in xgb_labels])

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
