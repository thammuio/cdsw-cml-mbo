import pandas as pd 
import numpy as np
import seaborn as sns


import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb

IS_LOCAL = False

import os

if(IS_LOCAL):
    PATH="data"
else:
    PATH="data"
print(os.listdir(PATH))

# Read
data_df = pd.read_csv(PATH+"/creditcard.csv")

# Check
print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])

## Glimpse
data_df.head()
data_df.describe()

## Check missing Data
total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

# ## Check unbalance
# temp = data_df["Class"].value_counts()
# df = pd.DataFrame({'Class': temp.index,'values': temp.values})

# trace = go.Bar(
#     x = df['Class'],y = df['values'],
#     name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",
#     marker=dict(color="Red"),
#     text=df['values']
# )
# data = [trace]
# layout = dict(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',
#           xaxis = dict(title = 'Class', showticklabels=True), 
#           yaxis = dict(title = 'Number of transactions'),
#           hovermode = 'closest',width=600
#          )
# fig = dict(data=data, layout=layout)
# iplot(fig, filename='class')