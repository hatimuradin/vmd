import time
from ngboost import NGBRegressor
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from evaluation import RMSE1, MAE1, MAPE1, R2

X = pd.read_csv('X.csv')
y = pd.read_csv('Y.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

t1=time.time()
ngb = NGBRegressor(
        # Dist=Normal,
        # Score=LogScore,
        # Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=2000,
        learning_rate=0.05,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.5,
        early_stopping_rounds=None,
).fit(X_train, y_train)
t2=time.time()

Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = RMSE1(y_test.to_numpy().flatten(), Y_preds)
print('Test RMSE', test_MSE)

# test MAE
test_MSE = MAE1(y_test.to_numpy().flatten(), Y_preds)
print('Test MAE', test_MSE)

# MAPE
test_MSE = MAPE1(y_test.to_numpy().flatten(), Y_preds)
print('Test MAPE', test_MSE)

# R2
test_MSE = R2(y_test.to_numpy().flatten(), Y_preds)
print('Test R2', test_MSE)

# Training Time
print('Training Time', t2-t1)