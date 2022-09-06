import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# Read the data
df_train = pd.read_csv('df_train_preprocessed.csv')
df_transported = df_train[['Transported']]

# Encoding the whole dataframe
d = defaultdict(LabelEncoder)
df_train = df_train.apply(lambda x: d[x.name].fit_transform(x))

# Create x and y variables x = train y = target
var_columns = [c for c in df_train.columns if c not in ['Transported', 'PassengerID', 'Name']]
X = df_train.loc[:, var_columns]
Y = df_train.loc[:, 'Transported']

# Split the test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

params={
 "learning_rate"    : [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 # "n_estimators"     : [2000,5000],
 "objective"        : ['binary:logistic'],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
 "booster"          : ['dart', 'gbtree']
# }

# params = {'n_estimators': np.arange(100,700),
#             'max_depth': np.arange(1,30),
#             'max_features': np.arange(2,10),
#             'min_samples_leaf': np.arange(1,20),
#             'min_samples_split': np.arange(2,20),
#             }

classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,Y)
timer(start_time) # timing ends here for "start_time" variable

print('best estimator:',random_search.best_estimator_)
print('best parameters:',random_search.best_params_)

# Create xgboost model by using different parameters (just trying for now)
# model_xgboost = xgboost.XGBClassifier(learning_rate=0.05,
#                                       max_depth=3,
#                                       n_estimators=5000,
#                                       subsample=0.5,
#                                       colsample_bytree=0.25,
#                                       eval_metric='auc',
#                                       booster='dart',
#                                       verbosity=1)
# Evaluation set which will be used to test
# eval_set = [(X_test, y_test)]
# Fit the xgboost
# model_xgboost.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# y_train_pred = model_xgboost.predict_proba(X_train)[:,1]
# y_test_pred = model_xgboost.predict_proba(X_test)[:,1]

# model_xgboost.save_model('XGB_titanic_spaceship_first_try.json')

# print("AUC Train: {:.4f}\nAUC Test: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                    # roc_auc_score(y_test, y_test_pred)))
