import pandas as pd
import numpy as np
import os

from sklearn import metrics
from restaurant.helper import bootstrap, train_val_split
import lightgbm as lgb
import matplotlib.pyplot as plt

# os.chdir('c:/users/kk/pycharmprojects/ml/restaurant')

params = {
    'boosting_type': 'gbdt',
    'metric': 'mape',
    'objective': 'quantile',
    'n_jobs': -1,
    'seed': 999,
    'learning_rate': 0.05,
    'bagging_fraction': 0.75,
    'bagging_freq': 10,
    'feature_fraction': 0.75,
    'bagging_seed': 999
}

# Prediction accuracy comparison on the validation data
# based on the model trained w/ or w/o outliers. 
# =======================================================

original_df = pd.read_csv('./data/train.csv')
score1 = []
score2 = []

for index in range(5000):

    # Data Load-in and split
    # ==========================
    train_df, val_df = train_val_split(original_df, 37, index)

    # Baseline Model
    # ==========================
    train_df = bootstrap(train_df, 80, 100, index)

    # We define the outliers for those with revenue >= 12500000
    train_trim_df = train_df[train_df.revenue < 12500000]

    X_train = train_df.copy().drop(['Id', 'Open Date', 'City', 'revenue', 'sample_number'], axis=1)
    X_train_trim = train_trim_df.copy().drop(['Id', 'Open Date', 'City', 'revenue', 'sample_number'], axis=1)
    X_val = val_df.copy().drop(['Id', 'Open Date', 'City', 'revenue'], axis=1)
    Y_train, Y_val, Y_train_trim = train_df['revenue'], val_df['revenue'], train_trim_df['revenue']

    for feat in list(X_train):
        X_train[feat] = X_train[feat].astype('category')
        X_val[feat] = X_val[feat].astype('category')
        X_train_trim[feat] = X_train_trim[feat].astype('category')

    reg = lgb.LGBMRegressor(**params)

    reg.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=100, early_stopping_rounds=100)
    pred_val1 = reg.predict(X_val, num_iteration=reg.best_iteration_)
    score1.append(np.sqrt(metrics.mean_squared_error(pred_val1, Y_val)))

    reg.fit(X_train_trim, Y_train_trim, eval_set=[(X_val, Y_val)], verbose=100, early_stopping_rounds=100)
    pred_val2 = reg.predict(X_val, num_iteration=reg.best_iteration_)
    score2.append(np.sqrt(metrics.mean_squared_error(pred_val2, Y_val)))

# Calculate average prediction error on the rolling basis.
score_df = pd.DataFrame(dict({'score1': score1, 'score2': score2}))
rolling_mean = score_df.cumsum().div(score_df.index.values+1, axis='index')

# Plot
# ==========================
plt.plot(range(len(score_df)), rolling_mean['score1'], c='black', label='w/ outliers')
plt.plot(range(len(score_df)), rolling_mean['score2'], c='blue', label='w/o outliers')
plt.xlabel('The number of trials')
plt.ylabel('Rolling average error(rmse) to the validation data')
plt.legend()
plt.title('Prediction error on the validation set based on a baseline lgbm w/ or w/o outliers')
