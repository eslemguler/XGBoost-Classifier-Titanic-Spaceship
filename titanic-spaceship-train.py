import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from joblib import dump, load


# Read the data
df_train = pd.read_csv('../df_train_preprocessed.csv')
df_transported = df_train[['Transported']]

# Encoding the whole dataframe
d = defaultdict(LabelEncoder)
df_train = df_train.apply(lambda x: d[x.name].fit_transform(x))

# Create x and y variables x = train y = target
var_columns = [c for c in df_train.columns if c not in ['Transported', 'PassengerID', 'Name']]
X = df_train.loc[:, var_columns]
y = df_train.loc[:, 'Transported']

# Split the test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create xgboost model by using different parameters
model_xgboost = xgboost.XGBClassifier(learning_rate=0.05,
                                      max_depth=7,
                                      min_child_weight =5,
                                      objective = 'binary:logistic',
                                      # n_estimators=300,
                                      subsample=0.5,
                                      colsample_bytree=0.3,
                                      # gamma = 0.1,
                                      eval_metric='auc',
                                      booster='gbtree',
                                      verbosity=1
                                      )

# This is another method I found online with hyperparameter optimization I got these values
# model_xgboost = xgboost.XGBClassifier(n_estimators=  179,
#                                         min_samples_split= 10,
#                                         min_samples_leaf= 1,
#                                         max_features= 5,
#                                         max_depth= 26,
#                                         )


# Evaluation set which will be used to test
eval_set = [(X_test, y_test)]
# Fit the xgboost
model_xgboost.fit(X_train, y_train, eval_set=eval_set, verbose=True)

y_train_pred = model_xgboost.predict_proba(X_train)[:,1]
y_test_pred = model_xgboost.predict_proba(X_test)[:,1]
dump(model_xgboost, '11.joblib')
