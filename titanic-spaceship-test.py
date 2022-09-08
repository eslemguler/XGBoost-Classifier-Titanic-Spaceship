import pandas as pd
import numpy as np
import math
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import json
from joblib import dump, load

test_df = pd.read_pickle('df_test_preprocess.pkl')

d = defaultdict(LabelEncoder)
test_df = test_df.apply(lambda x: d[x.name].fit_transform(x))

var_columns = [c for c in test_df.columns if c not in ['Transported', 'PassengerID', 'Name']]
X = test_df.loc[:, var_columns]

model = load('xgboost_model.joblib')
pred = model.predict(X)

submission = pd.DataFrame()
submission['Transported'] = pred
submission.to_csv('submission.csv', index=False)
