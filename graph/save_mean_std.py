# 완료
"""
save_mean_std.py
==================
calculate mean and standard deviation of samples for each person
these values are used for 'immersion_probability.py'
"""
import pandas as pd
data = pd.read_csv('../dataset/exp1+2_raw.csv', delimiter=',')
num_person = 30
# user = range(1,num_person+1)
user = [6]
for i in user:
    norm_data = pd.DataFrame(columns=data.columns)
    user_data = data[data["User Code"].isin([i])]
    new_df = pd.DataFrame(columns=data.columns)
    new_df.loc['mean'] = user_data.mean(axis=0)
    new_df.loc['std'] = user_data.std(axis=0)
    pd.set_option('display.expand_frame_repr', False)  # Dataframe 생략없이 모두 출력하기
    new_df.to_csv('./sample_data/user%d_mean_std.csv' % i)