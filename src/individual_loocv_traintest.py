# 완료 - 하지만 확인 한 번 더 하기
'''
individual_loocv.py
=========================
individual model (person-dependent model) with Leave-one-instance-out cross-validation
train/valid/test version
'''
import os
import warnings
import joblib
import pandas as pd
import numpy as np
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

def individual_model_traintest(data_path, fs, load_fs, print_test, load_pretrained_model, save_dir, log_path, seed):

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
    # data = data[data["User Code"] != 26]
    num_person = len(data['User Code'].unique())    # the number of population
    num_instance = len(data['Video Code'].unique()) # the number of instance(vides) = number of folds
    user_list = data['User Code'].unique().tolist()
    instance_list = data['Video Code'].unique().tolist()

    # Prepare data frames to print the result of models
    # index: fold number, columns: classifier name
    total_train_df = pd.DataFrame(index=range(1, num_person+1), columns=clf_names)  # train result
    # total_valid_df = pd.DataFrame(index=range(1, num_person+1), columns=clf_names)  # valid result
    total_test_df = pd.DataFrame(index=range(1, num_person+1), columns=clf_names)  # test result
    if fs: total_fs_df = pd.DataFrame(index=range(1, num_person+1), columns=all_features)  # feature selection result
    

    ''' (2) individual modeling '''

    # Leave-one-instance-out (LOOCV) cross-validation
    out_put(f'Run a total of {num_instance} folds for each user', log_path)
    for i in user_list:
        train_df = pd.DataFrame(index=range(0, num_instance), columns=clf_names)
        # valid_df = pd.DataFrame(index=range(0, num_instance), columns=clf_names)
        test_df = pd.DataFrame(index=range(0, num_instance), columns=clf_names)
        fs_df = pd.DataFrame(index=range(0, num_instance), columns=all_features)

        out_put("\nIndividual model - User %2d " % (i), log_path)
        user_data = data[data["User Code"].isin([i])]

        # if save_on:
        #     model_path = os.path.join(model_save_path, 'P{0}'.format(i))
        #     if not os.path.exists(model_path):
        #         os.mkdir(model_path)
        #     f = open(os.path.join(model_path, 'feature_list.txt'), 'w')

        for j in range(num_instance):
            out_put("******************** [ %2d fold ] ********************" % (j + 1), log_path)

            test_video = instance_list[num_instance-j-1]

            test_data = user_data[user_data["Video Code"].isin([test_video])]  
            train_data = user_data[~user_data["Video Code"].isin([test_video])]
            out_put(f"test video : {test_video}", log_path)

            x_train, y_train, x_test, y_test, columns = construct_dataset(train_df=train_data, valid_df=test_data, mode='individual')

            # Feature selection
            if fs:
                out_put(f"Shape of features used (before FS): {x_train.shape}, {x_test.shape}", log_path)

                # Run feature selection
                if load_fs:
                    feature_list = np.load(f'pretrained_model/individual_top10/feature/P{i}/{j+1}fold_feature_list.npy')
                else:
                    feature_list, mask = select_features(x_train, y_train, 'rfe', 10)
                    fs_df.iloc[j] = mask
                    os.makedirs(f'{save_dir}/feature/P{i}', exist_ok=True)
                    np.save(f'{save_dir}/feature/P{i}/{j+1}fold_feature_list.npy', feature_list)
                
                # Reconstruct data with selected features
                x_train = x_train[feature_list]
                if print_test: x_test = x_test[feature_list]
                out_put(f"Shape of features used (after FS): {x_train.shape}, {x_test.shape}", log_path)

            else:
                out_put(f"Shape of features used: {x_train.shape}, {x_test.shape}", log_path)

            # Classification
            for name, clf in zip(clf_names, classifiers):
                # Load model
                if load_pretrained_model: clone_clf = joblib.load(f'pretrained_model/individual_top10/model/P{i}/{j+1}fold_{name}.pkl')
                else: clone_clf = clone(clf)
                
                # Training
                history = clone_clf.fit(x_train, y_train)

                # Train accuracy
                score = clone_clf.score(x_train, y_train)
                score = round(score * 100, 2)
                train_df[name].iloc[j] = score

                # Test accuarcy
                if print_test:
                    t_score = clone_clf.score(x_test, y_test)
                    t_score = round(t_score * 100, 2)
                    test_df[name].iloc[j] = t_score

                # Save model
                os.makedirs(f'{save_dir}/model/P{i}', exist_ok=True)
                filename = f'{save_dir}/model/P{i}/{j+1}fold_{name}.pkl'
                joblib.dump(clone_clf, filename)

        # Calculate mean score of one person model
        train_df.loc['mean'] = train_df.mean(axis=0)
        if print_test: test_df.loc['mean'] = test_df.mean(axis=0)
        if fs and not load_fs: fs_df.loc['sum'] = fs_df.sum(axis=0)

        out_put("\nTRAIN ACCURACY", log_path)
        out_put(train_df, log_path)
        if print_test:
            out_put("\nTEST ACCURACY", log_path)
            out_put(test_df, log_path)
        if fs and not load_fs:
            out_put("\nFEATURE SELECTION RESULT", log_path)
            out_put(fs_df, log_path)
            
        # Add mean score to total accuracy dataframe
        total_train_df.loc[i] = train_df.loc['mean']
        if print_test: total_test_df.loc[i] = test_df.loc['mean']
        if fs and not load_fs:  total_fs_df.loc[i] = fs_df.loc['sum']
    
    total_train_df.loc['acc'] = total_train_df.mean(axis=0)
    if print_test: total_test_df.loc['acc'] = total_test_df.mean(axis=0)
    if fs and not load_fs: total_fs_df.loc['sum'] = total_fs_df.sum(axis=0)
    
    out_put("\n---- Accuracy of all individual models ----", log_path)
    out_put("\nTRAIN ACCURACY", log_path)
    out_put(total_train_df, log_path)
    if print_test:
        out_put("\nTEST ACCURACY", log_path)
        out_put(total_test_df, log_path)
    if fs and not load_fs:
        out_put("\nFEATURE SELECTION RESULT", log_path)
        out_put(total_fs_df, log_path)
        total_fs_df.to_csv(f'{save_dir}/feature_selection.csv')
