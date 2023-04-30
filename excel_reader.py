import pandas as pd
# from vmdpy import VMD 
from support_VMD import VMD
import scipy.io

dataset = pd.read_csv('PRSA_Data_1.csv')
mat = scipy.io.loadmat('eeg.mat')

PM10 = list(dataset.loc[:, 'PM10'].values)


# some sample parameters for VMD
alpha = 2000       # moderate bandwidth constraint
tau = 0.            # noise-tolerance (no strict fidelity enforcement)
K = 4              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-6


# result = VMD(PM10, alpha, tau, K, DC, init, tol)
result = VMD(PM10, K)
pass