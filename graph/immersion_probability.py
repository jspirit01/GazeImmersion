# 완료
"""
immersion_probability.py
=========================
Draw the line graph with immersion probability as y-axis and time as x-axis
"""
import os

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange

num_person = 30
num_video = 14


def segment_processing(userCode, videoCode):
    """
    load recording segments of the user and conduct pre-processing

    :param userCode: user code
    :param videoCode: video code
    :return: pre-processed segment data
    """
    v_type = 1  # video type : exp1 or exp2
    if videoCode > 6:
        v_type = 2
        video = videoCode - 6
    else:
        video = videoCode

    # load the segment data of the user
    # data_path = './sample_data/exp{0}_10sec/segment_features_user{1}.csv'.format(v_type, userCode)
    data_path = 'graph/sample_data/segment_features_user{1}.csv'.format(v_type, userCode)  
    df = pd.read_csv(data_path)

    # extract the video segments that you want to draw the graph
    seg_list = df['Unnamed: 0']
    idx_list = []
    for idx, seg_name in enumerate(seg_list):
        video_num, seg_idx = seg_name.split('_seg')
        seg_idx = seg_idx.zfill(2)
        seg_rename = video_num + '_seg' + seg_idx
        df['Unnamed: 0'].iloc[idx] = seg_rename

        if (str(video) + '_') in seg_rename:
            idx_list.append(idx)
    df = df.iloc[idx_list]
    df = df.sort_values(by='Unnamed: 0')

    for col in df.columns:
        if '_count' in col:
            df = df.drop(col, axis=1)

    # load mean and standard deviation value of the user to normalize segments.
    norm_path = 'graph/sample_data/user%d_mean_std.csv' % userCode
    norm_df = pd.read_csv(norm_path, index_col=0)

    # normalize segments
    for col in df.columns[1:]:
        mean = norm_df[col].loc['mean']
        std = norm_df[col].loc['std']
        df[col] = (df[col] - mean) / std

    # pre-processing
    test_df = df.drop(['Unnamed: 0'], axis=1)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    x_test = test_df.fillna(0)

    return x_test


def predict_probability(m_type, x_test, clf):
    """
    predict immersion probability from pre-trained classification model

    :param m_type: model type (population or individual)
    :param x_test: segment data of the user
    :param clf: classifier used to predict the label
    :return: the values of immersion probabilities as y-axis values
    """
    # set model path to be loaded
    # it will use the model trained with top-10 features for better accuracy.
    if m_type == 'population':
        model_root = 'model/' + m_type + '/top10'
        num_fold = num_person
    else:
        model_root = 'model/' + m_type + '/P' + str(userCode)
        num_fold = num_video

    # load the list of feature selection
    if not os.path.exists(os.path.join(model_root, 'feature_list.txt')):
        print("Can't read feature list. Please check the model type and make sure the file exists")
        print(os.path.abspath(os.path.join(model_root, 'feature_list.txt')))
        exit(0)
    with open(os.path.join(model_root, 'feature_list.txt'.format(userCode)), 'r') as f:
        feature_lines = f.readlines()

    # predict immersion probability given segemnts
    prob = None
    for i, feature_list in zip(trange(0, num_fold), feature_lines):

        # construct data with feature list
        feature_list = feature_list.rstrip('\n')
        feature_list = feature_list.split(',')
        x_test_fold = x_test[feature_list]

        # load the pre-trained model
        model_path = os.path.join(model_root, '{0}fold_{1}.pkl'.format(i + 1, clf))
        load_clf = joblib.load(model_path)

        # predict immersion probability
        y_predict = load_clf.predict_proba(x_test_fold)
        if prob is None:
            prob = y_predict
        else:
            prob += y_predict

    # average the probability for all folds
    y_values = []
    probs = prob / num_person
    for i, seg_prob in enumerate(probs):
        immersive_prob = seg_prob[1] * 100  # immersive일 확률
        y_values.append([i * 10, immersive_prob])

    return y_values


def draw_graph(userCode, videoCode, m_type='population', clf='RandomForest'):
    """
    draw immersion probability graph when the user watches the video.

    :param userCode: user code
    :param videoCode: video code
    :param m_type: model type (population or individual)
    :param clf: classifier used to predict the label
    :return: (no return)
    """

    print("Participant {0} - Video {1}".format(userCode, videoCode))

    # load recording segments of the user and conduct pre-processing
    print("load the data ...")
    x_test = segment_processing(userCode, videoCode)

    # predict immersion probability given segemnts
    print("calculate the probability ...")
    y_values = predict_probability(m_type, x_test, clf)

    # plot the graph
    print("save the graph ...")
    data = pd.DataFrame(y_values, columns=['time (second)', 'immersion probability (%)'])

    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    g = sns.lineplot(x='time (second)', y='immersion probability (%)', data=data, )
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.axhline(y=50, color='r', linewidth=1)
    # plt.xticks(np.arange(0, data['time (second)'].iloc[-1], 20), fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('time (second)', fontsize=24)
    plt.ylabel('immersion proability (%)', fontsize=24)
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    graph_path = 'graph/result/'
    plt.savefig(graph_path + 'P%d_V%d.png' % (userCode, videoCode))
    # data.to_csv(graph_path + 'P%d_V%d.csv' % (userCode, videoCode))

    plt.clf()

    print("The immersion graph is successfully saved!")
    print("path:", os.path.abspath(graph_path) + '\\P%d_V%d' % (userCode, videoCode))


userCode = 20    # choose 1~30
videoCode = 3   # choose 1~14
draw_graph(userCode=userCode, videoCode=videoCode)
