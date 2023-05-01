import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


X = pd.read_csv('X.csv')
y = pd.read_csv('Y.csv')

# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }
                        
# instantiate the classifier 
xgb_clf = XGBClassifier(**params)

le = LabelEncoder()
y_train = le.fit_transform(y_train)

# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)

# alternatively view the parameters of the xgb trained model
print(xgb_clf)

# make predictions on test data
y_pred = xgb_clf.predict(X_test)

# check accuracy score
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# save predicted values to file
np.savetxt('predicted_PM10.csv', y_pred, delimiter=",")
