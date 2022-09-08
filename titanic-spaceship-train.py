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
df_train = pd.read_pickle('df_train_preprocess.pkl')

# Encoding the whole dataframe because we will use classifier
d = defaultdict(LabelEncoder)
df_train = df_train.apply(lambda x: d[x.name].fit_transform(x))

# Create x and y variables x = train y = target
var_columns = [c for c in df_train.columns if c not in ['Transported', 'PassengerID', 'Name']]
X = df_train.loc[:, var_columns]
y = df_train.loc[:, 'Transported']

# Split the test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create xgboost model by using different parameters
# You can get these parameters from hyperparameter_tuning script
model_xgboost = xgboost.XGBClassifier(learning_rate=0.075,
                                      max_depth=7,
                                      min_child_weight =5,
                                      objective = 'binary:logistic',
                                      n_estimators=400,
                                      subsample=0.4,
                                      colsample_bytree=0.6,
                                      gamma = 0.1,
                                      eval_metric='auc',
                                      booster='gbtree',
                                      verbosity=1
                                      )

# Evaluation set which will be used to test
eval_set = [(X_test, y_test)]
# Fit the xgboost
model_xgboost.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Save the model
dump(model_xgboost, 'xgboost_model.joblib')
