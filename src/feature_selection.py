'''
feature_selection.py
========================
use it for feature selection
In our paper, we use RFE to select top 10 features
'''
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC


def select_features(x, y=None, tag='kbest', k=10):
    print("Feature Selection ... ")
    if tag == 'kbest':
        fs_model = SelectKBest(f_classif, k)
        fs_model = fs_model.fit(x,y)
    elif tag == 'tree':
        fs_model = ExtraTreesClassifier()
        fs_model = fs_model.fit(x, y)
    elif tag == 'rfe':
        model_RF = RandomForestClassifier()
        fs_model = RFE(model_RF, n_features_to_select=k)
        fs_model = fs_model.fit(x, y)
    else:
        return -1

    new_features = []
    mask = fs_model.get_support()
    for bool, feature in zip(mask, x.columns):
        if bool:
            new_features.append(feature)

    return new_features, mask*1