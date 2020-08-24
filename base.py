import pandas as pd
import numpy as np
import os

from sklearn import metrics
from restaurant.helper import bootstrap
import lightgbm as lgb
import matplotlib.pyplot as plt

# os.chdir('c:/users/kk/pycharmprojects/ml/restaurant')

# Data Load-in and split
# ==========================
train_df = pd.read_csv('./data/train.csv')
val_df = train_df.sample(n=37, random_state=999)
train_df = train_df.drop(val_df.index, axis=0)
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# Baseline Model
# ==========================
train_df = bootstrap(train_df, 80, 100)
X_train = train_df.copy().drop(['Id', 'Open Date', 'City', 'revenue', 'sample_number'], axis=1)
X_val = val_df.copy().drop(['Id', 'Open Date', 'City', 'revenue'], axis=1)
Y_train, Y_val = train_df['revenue'], val_df['revenue']

params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'n_jobs': -1,
    'seed': 999,
    'learning_rate': 0.05,
    'bagging_fraction': 0.75,
    'bagging_freq': 10,
    'feature_fraction': 0.75,
}

for feat in list(X_train):
    X_train[feat] = X_train[feat].astype('category')
    X_val[feat] = X_val[feat].astype('category')

df_lgb = lgb.Dataset(data=X_train, label=Y_train, feature_name=list(X_train))
comp = lgb.cv(params, train_set=df_lgb, num_boost_round=1000, nfold=10, stratified=False, early_stopping_rounds=100)

reg = lgb.LGBMRegressor(**params)
reg.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=100, early_stopping_rounds=100)

pred_val = reg.predict(X_val, num_iteration=reg.best_iteration_)
score = np.sqrt(metrics.mean_squared_error(pred_val, Y_val))

# Plotting the prediction results 
# ==========================
plt.scatter(range(len(Y_val)), Y_val, c='black', label='validation data')
plt.scatter(range(len(Y_val)), pred_val, c='blue', label='prediction')
plt.xlabel('validation data index')
plt.ylabel('revenue')
plt.minorticks_on()
plt.grid(which='minor', axis='x', linestyle=':')
plt.grid(which='major', axis='x', linestyle=':')
plt.legend()
plt.title('Prediction on the validation set based on a baseline lgbm, rmse: ' + str(int(score)))
