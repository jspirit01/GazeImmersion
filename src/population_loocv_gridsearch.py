"""
population_loccv_gridsearch.py
=========================
gridsearch로 최적의 model parameter 찾기
"""

from tkinter import Grid
import warnings
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    accuracy_score
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from feature_selection import select_features
from feature_preprocessing import construct_dataset, shuffle_data

warnings.filterwarnings('ignore')

num_person = 30  # the number of population

# leave-one-person-out cross-validation

df = pd.read_csv('dataset/exp1+2_usernorm.csv', delimiter=',')
groups = df['User Code'].to_list()

# remove unnecessary columns
unnecessary_cols = ["User Code", "Video Code", "total_survey",
                    "captivation", "cap_I/N", "dissociation", "dis_I/N",
                    "comprehension", "com_I/N", "transportation", "tra_I/N"]
df = df.drop(unnecessary_cols, axis=1)

# split data and label
x = df.drop("total_I/N", axis=1)
y = np.ravel(df['total_I/N'])

gkf = GroupKFold(n_splits=num_person)   # leave one person out data generator

# svc
# param_grid = {
#     "kernel" : ['linear', 'poly', 'rbf',],
#     "C": [0.0001,0.001,0.01,0.1,1.10,100,1000],
#               "gamma": [0.001, 0.001, 0.01, 0.1, 1,10,100,1000]}

# grid_search = RandomizedSearchCV(estimator=SVC(),
#                        param_grid=param_grid,
#                        cv=gkf,
#                        verbose=2,
#                        n_jobs=-1).fit(x, y, groups=groups)

# DT
# param_grid = {
#     'max_depth':[2,3,4,5, None], 
#     'min_samples_split':[2,3,4,5],
#     'random_state': [42]
#     }
# grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),
#                        param_grid=param_grid,
#                        cv=gkf,
#                        verbose=3,
#                        n_jobs=-1).fit(x, y, groups=groups)

#knn
# param_grid = {
#     'n_neighbors' : list(range(1,30)),
#     'weights' : ["uniform", "distance"],
#     'metric' : ['euclidean', 'manhattan', 'minkowski']
#     }
# grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
#                        param_grid=param_grid,
#                        cv=gkf,
#                        verbose=3,
#                        n_jobs=-1).fit(x, y, groups=groups)

# NB
# param_grid = {
#     'alpha': [0.01, 0.1, 0.5, 1.0, 10.0], #BernoulliNB
#     # 'var_smoothing': np.logspace(0,-9, num=100) #GaussianNB
# }
# grid_search = GridSearchCV(estimator=BernoulliNB(),
#                        param_grid=param_grid,
#                        cv=gkf,
#                        verbose=1,
#                        scoring='accuracy',
#                        n_jobs=-1).fit(x, y, groups=groups)


#LR
# param_grid = {
#     'C' : [0.001, 0.01, 0.1, 1, 10, 100],
#     'max_iter' : [100,1000, 10000],
#     'solver': ['newton-cg', 'liblinear'],
#     'class_weight': ['balanced', 'None'],
#     'penalty': ['l1'],
#     'random_state': [42]
# }
# grid_search = GridSearchCV(estimator=LogisticRegression(),
#                        param_grid=param_grid,
#                        cv=gkf,
#                        verbose=1,
#                        scoring='accuracy',
#                        n_jobs=-1).fit(x, y, groups=groups)

#Ada
# param_grid = {
#     'base_estimator__max_depth':[2,3,4],
#     'n_estimators':[10,50,250,1000],
#     'learning_rate':[0.01,0.1]}

# grid_search = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
#                        param_grid=param_grid,
#                        cv=gkf,
#                        verbose=1,
#                        scoring='accuracy',
#                        n_jobs=-1).fit(x, y, groups=groups)

# RF
param_grid = {
    'max_depth' : [3,4,5,6,7,8] ,
    'n_estimators': [50, 100, 200], 
    'min_samples_leaf':[1,2,3,4,],
    'min_samples_split':[2,3,4,5,8,16,20],
    'n_jobs': [-1]}

grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                       param_grid=param_grid,
                       cv=gkf,
                       verbose=1,
                       scoring='accuracy',
                       n_jobs=-1).fit(x, y, groups=groups)


print(grid_search.best_params_)
print(grid_search.best_score_)