'''
feature_preprocessing.py
===========================
functions for feature pre-processing

revision log
- 2023-09-23
'''
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

def construct_dataset(train_df, valid_df, test_df=None, mode='population'):
    # Shuffle samples
    train_df, valid_df, test_df = shuffle_data(train_df, valid_df, test_df)
    
    # Remove unncecessary columns
    if mode == 'population':
        unnecessary_cols = ["User Code", "Video Code", "total_survey",
                            "captivation", "cap_I/N", "dissociation", "dis_I/N",
                            "comprehension", "com_I/N", "transportation", "tra_I/N"]
    elif mode == 'individual':
        unnecessary_cols = ["User Code", "Video Code", "total_survey"]       
    else:
        raise Exception("mode should be 'population' or 'indivdiaul'!")
    train_df = train_df.drop(unnecessary_cols, axis=1)
    valid_df = valid_df.drop(unnecessary_cols, axis=1)
    if isinstance(test_df, pd.DataFrame):
        test_df = test_df.drop(unnecessary_cols, axis=1)

    # Split sample into data and label
    x_train = train_df.drop("total_I/N", axis=1)
    y_train = np.ravel(train_df['total_I/N'])
    x_valid = valid_df.drop("total_I/N", axis=1)
    y_valid = np.ravel(valid_df['total_I/N'])
    if isinstance(test_df, pd.DataFrame):
        x_test = test_df.drop("total_I/N", axis=1)
        y_test = np.ravel(test_df['total_I/N'])

    # get columes of data
    columns = x_train.columns

    if isinstance(test_df, pd.DataFrame):
        return x_train, y_train, x_valid, y_valid, x_test, y_test, columns
    else:
        return x_train, y_train, x_valid, y_valid, columns
    

def shuffle_data(train_df, valid_df, test_df=None, random_state=0):
    train_df = shuffle(train_df, random_state=random_state)
    valid_df = shuffle(valid_df, random_state=random_state)
    if not isinstance(test_df, pd.DataFrame):
        return train_df, valid_df, None
    else:
        test_df = shuffle(test_df, random_state=random_state)
        return train_df, valid_df, test_df
        
