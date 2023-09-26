# 완료
"""
narmalize_per_person.py
==========================
normalize samples per person with z-score
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# load raw data
data = pd.read_csv('../dataset/exp1+2_raw.csv', delimiter=',')

# normalize data per person
num_person = 30
for i in range(1, num_person+1):
    print("Normalize samples of User %d" % i)

    user_data = data[data["User Code"].isin([i])]
    values = user_data.iloc[:, 2:51]    # column range : FD_mean ~ PD_max

    norm_values = StandardScaler().fit_transform(values)
    user_data.iloc[:, 2:51] = norm_values

    user_data.to_csv('../dataset/user/user%s_norm.csv' % str(i), index=False)

# concat the result of each person
dataset = None
for i in range(1, num_person + 1):
    data = pd.read_csv('../dataset/user/user%s_norm.csv' % str(i), delimiter=',')
    if i == 1:
        dataset = data
        continue
    dataset = pd.concat([dataset, data], axis=0)

dataset.to_csv('../dataset/exp1+2_usernorm.csv', index=False)


