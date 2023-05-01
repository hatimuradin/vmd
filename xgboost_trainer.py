import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb 

vmd_signals = pd.read_csv('decomposed_data.csv')

# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=X,label=y)



