'''
population_loocv_traintest.py
=========================
population model with Leave-one-person-out cross-validation
train/test version
'''

import warnings
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    accuracy_score
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from src.feature_selection import select_features
from src.feature_preprocessing import construct_dataset, shuffle_data
from src.utils import out_put
# warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)

def population_model_traintest(data_path, fs, load_fs, print_test, load_pretrained_model, save_dir, log_path, seed):
    ''' (1) initialization before modeling '''

    first_log = f"""
    fs: {fs},
    load_fs: {load_fs}, 
    print_test: {print_test}, 
    load_pretrained_model: {load_pretrained_model}, 
    save_dir: {save_dir},
    seed: {seed}
    """
    out_put(first_log, log_path)

    os.makedirs(f'{save_dir}/feature/', exist_ok=True)
    os.makedirs(f'{save_dir}/model/', exist_ok=True)

    # List of all gaze features (count : 49)
    all_features = ['FD_mean', 'FD_std', 'FD_min', 'FD_1st', 'FD_median', 'FD_3rd', 'FD_max',
                    'SD_mean', 'SD_std', 'SD_min', 'SD_1st', 'SD_median', 'SD_3rd', 'SD_max',
                    'FR_mean', 'FR_std', 'FR_min', 'FR_1st', 'FR_median', 'FR_3rd', 'FR_max',
                    'SR_mean', 'SR_std', 'SR_min', 'SR_1st', 'SR_median', 'SR_3rd', 'SR_max',
                    'SV_mean', 'SV_std', 'SV_min', 'SV_1st', 'SV_median', 'SV_3rd', 'SV_max',
                    'SA_mean', 'SA_std', 'SA_min', 'SA_1st', 'SA_median', 'SA_3rd', 'SA_max',
                    'PD_mean', 'PD_std', 'PD_min', 'PD_1st', 'PD_median', 'PD_3rd', 'PD_max']

    # Prepare classifiers
    base_model = DummyClassifier(strategy='most_frequent', random_state=0)  # ZeroR
    svc_model = SVC(kernel='rbf', C=1.0)  # SVM
    knn_model = KNeighborsClassifier(n_neighbors=7)  # k-Nearest Neighbors
    lr_model = LogisticRegression(C=1, random_state=0, max_iter=10000, solver='newton-cg',
                                class_weight="balanced")  # Logistic Regression
    dt_model = DecisionTreeClassifier(max_depth=3)  # Decision Tree
    rf_model = RandomForestClassifier(criterion='entropy', n_estimators=100, n_jobs=2, random_state=0)  # Random Forest
    ab_model = AdaBoostClassifier()  # AdaBoost
    nb_model = BernoulliNB()  # Naive Bayse
    clf_names = ['ZeroR', 'DecisionTree', 'kNN', 'NaiveBayes', 'SVM', 'LogisticRegression', 'AdaBoost', 'RandomForest']
    classifiers = [base_model, dt_model, knn_model, nb_model, svc_model, lr_model, ab_model, rf_model]


    # Load dataset
    data = pd.read_csv(data_path)
    data = data[data["User Code"] != 26]
    num_person = len(data['User Code'].unique())    # the number of population = number of folds
    user_list = data['User Code'].unique().tolist()
    
    # Prepare data frames to print the result of models
    # index: fold number , columns: classifier name
    train_df = pd.DataFrame(index=range(1, num_person+1), columns=clf_names)  # train result
    # valid_df = pd.DataFrame(index=range(1, num_person+1), columns=clf_names)  # valid result
    test_df = pd.DataFrame(index=range(1, num_person+1), columns=clf_names)  # test result
    if fs: fs_df = pd.DataFrame(index=range(1, num_person+1), columns=all_features)  # feature selection result
    

    ''' (2) population modeling '''

    # Leave-one-person-out (LOOCV) cross-validation
    out_put(f'Run a total of {num_person} folds.', log_path)
    for i in range(0, num_person):
        out_put("******************** [ %2d fold ] ********************" % (i + 1), log_path)

        test_user = user_list[num_person-i-1]
        # valid_user = user_list[num_person-i-2]
        # test_user = num_person - i
        # valid_user = num_person - i - 1
        # if valid_user <= 0: valid_user = num_person

        test_data = data[data["User Code"].isin([test_user])]  # not yet use
        # valid_data = data[data["User Code"].isin([valid_user])]
        train_data = data[~data["User Code"].isin([test_user])]
        out_put(f"test user : {test_data['User Code'].iloc[0]}", log_path)

        x_train, y_train, x_test, y_test, columns = construct_dataset(train_df=train_data, valid_df=test_data, mode='population')
        
        # Feature selection
        if fs:
            out_put(f"Shape of features used (before FS): {x_train.shape}, {x_test.shape}", log_path)
            
            # Run feature selection
            if load_fs:
                feature_list = np.load(f'pretrained_model/populatin_top10/feature/{i+1}fold_feature_list.npy')
            else:
                feature_list, mask = select_features(x_train, y_train, 'rfe', 10)
                fs_df.iloc[i] = mask
                np.save(f'{save_dir}/feature/{i+1}fold_feature_list.npy', feature_list)                
                
            # Reconstruct data with selected features
            
            x_train = x_train[feature_list]
            # x_valid = x_valid[feature_list]
            if print_test: x_test = x_test[feature_list]
            out_put(f"Shape of features used (after FS): {x_train.shape}, {x_test.shape}", log_path)

        else:
            out_put(f"Shape of features used: {x_train.shape}, {x_test.shape}", log_path)

        # Classification
        for name, clf in zip(clf_names, classifiers):
            # Load model
            if load_pretrained_model: clone_clf = joblib.load(f'pretrained_model/population_top10/model/{i+1}fold_{name}.pkl')
            else: clone_clf = clone(clf)
            
            # Training
           
           
            history = clone_clf.fit(x_train, y_train)

            # Train accuracy
            score = clone_clf.score(x_train, y_train)
            score = round(score * 100, 2)
            train_df[name].iloc[i] = score

            # Valid accuracy
            # v_score = clone_clf.score(x_valid, y_valid)
            # v_score = round(v_score * 100, 2)
            # valid_df[name].iloc[i] = v_score

            # Test accuracy 
            if print_test:    
                t_score = clone_clf.score(x_test, y_test)
                t_score = round(t_score * 100, 2)
                test_df[name].iloc[i] = t_score
                
            # Save model
            filename = f'{save_dir}/model/{i+1}fold_{name}.pkl'
            joblib.dump(clone_clf, filename)

    # Print the results
    train_df.loc['acc'] = train_df.mean(axis=0)
    # valid_df.loc['acc'] = valid_df.mean(axis=0)
    if print_test: test_df.loc['acc'] = test_df.mean(axis=0)
    if fs and not load_fs: fs_df.loc['sum'] = fs_df.sum(axis=0)
    
    out_put("\nTRAIN ACCURACY", log_path)
    out_put(train_df, log_path)
    # out_put("\nVALIDATION ACCURACY", log_path)
    # out_put(valid_df, log_path)
    if print_test:
        out_put("\nTEST ACCURACY", log_path)
        out_put(test_df, log_path)
    if fs and not load_fs:
        out_put("\nFEATURE SELECTION RESULT", log_path)
        out_put(fs_df, log_path)
        fs_df.to_csv(f'{save_dir}/feature_selection.csv')
