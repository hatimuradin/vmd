import pandas as pd
import sys
import numpy as np
import math
from support_VMD import VMD
from sklearn.preprocessing import StandardScaler




def data_interval(data, itv):

    data_itved = []
    for i in range(len(data)):
        if (i % itv) == 0:
            data_itved.append(data[i])

    return data_itved

def create_data(data, train_num, time_step):
    TS_X = []

    for i in range(data.shape[0] - time_step):
        b = data[i:(i + time_step), 0]
        TS_X.append(b)

    # dataX1 = TS_X[:train_num]
    # dataX2 = TS_X[train_num:]
    # dataY1 = data[time_step: train_num + time_step, 0]
    # dataY2 = data[train_num + time_step:, 0]

    dataX1 = TS_X[:train_num]
    dataX2 = TS_X[train_num:]
    dataY1 = data[time_step: train_num + time_step, 0]
    dataY2 = data[train_num + time_step:, 0]

    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)

def data_check_test(data):

    # data_back = np.array(data).T.tolist()
    data_checked = np.array(data).T.tolist()

    for i in range(len(data_checked)):
        data_feature = data_checked[i]
        for j in range(len(data_feature)):
            NaN_judge = math.isnan(data_feature[j])
            if NaN_judge is True or data_feature[j] == 0:
                if j == 0:  # mean of first.
                    h1 = 1
                    while math.isnan(data_checked[i][j + h1]) is True:
                        h1 = h1 + 1
                    mean = data_checked[i][j + h1]
                elif j == len(data) - 1:  # mean of last.
                    h2 = - 1
                    while math.isnan(data_checked[i][j + h2]) is True:
                        h2 = h2 - 1
                    mean = data_checked[i][j + h2]
                else:
                    h1 = 1
                    while math.isnan(data_checked[i][j + h1]) is True:
                        h1 = h1 + 1
                    h2 = - 1
                    while math.isnan(data_checked[i][j + h2]) is True:
                        h2 = h2 - 1
                    mean = (data_checked[i][j + h1] + data_checked[i][j + h2]) / 2
                data_feature[j] = mean
        data_checked[i] = data_feature

    data_checked = np.array(data_checked).T.tolist()

    return data_checked


def read_csv_PM10(filename_list, trainNum, testNum, startNum, file_num, interval):

    targetData = None

    for i in range(file_num):
        filename = filename_list[0] + str(i + 1) + filename_list[1]
        dataset = pd.read_csv(filename, encoding='gbk')

        # 输入数据
        file_PM2 = dataset.iloc[:, 5]  # PM2.5 2k
        file_PM10 = dataset.iloc[:, 6]   # PM10
        file_SO2 = dataset.iloc[:, 7]   # SO2 2k
        file_NO2 = dataset.iloc[:, 8]
        file_CO = dataset.iloc[:, 9]
        file_O3 = dataset.iloc[:, 10]
        file_TEMP = dataset.iloc[:, 11]
        file_PRES = dataset.iloc[:, 12]
        file_DEWP = dataset.iloc[:, 13]

        # 预测对象
        file_targetData = file_PM10
        file_targetData = np.array(file_targetData).reshape(-1, 1)

        if i == 0:
            targetData = file_targetData
        else:
            targetData = np.r_[targetData, file_targetData]

    all_data = targetData
    all_data = all_data.tolist()
    # all_data_checked = data_check(all_data)
    all_data_checked = data_check_test(all_data)
    all_data_checked = data_interval(all_data_checked, interval)
    all_data_checked = np.array(all_data_checked)

    print("Total Data:", all_data_checked.shape)
    data_shape = all_data_checked.shape
    max_dataset = data_shape[0]
    if max_dataset < (startNum + testNum + trainNum):
        print('This Dataset is not enough for such trainNum & testNum.')
        sys.exit(1)

    return all_data_checked.reshape(-1, )    
    
def load_data_VMD(data, K):
    print('VMD_data loading.')

    targetData = np.array(data).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    # testY = targetData[trainNum: (trainNum + testNum), :]

    imf_list = VMD(targetData.reshape(-1, ), K)
    imf_list = imf_list.tolist()

    return imf_list


filename1 = "PRSA_Data_"
filename2 = ".csv"
filename = [filename1, filename2]
# training number
startNum = 0
interval = 1
trainNum = 35064
testNum = 0
K = 4
dataset = read_csv_PM10(filename, trainNum, testNum, startNum, 1, interval)
np.savetxt("Y.csv", dataset, delimiter=",")
VMD_list = load_data_VMD(dataset, K)
transposed_VMD = list(zip(*VMD_list))
np.savetxt("X.csv", transposed_VMD, delimiter=",")

