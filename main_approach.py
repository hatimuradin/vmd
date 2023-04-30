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
    
def load_data_VMD(trainNum, testNum, startNum, data):
    print('VMD_data loading.')
    wvlt_lv = 8
    # all_data_checked = data
    targetData = data

    # 处理预测信息，划分训练集和测试集
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    testY = targetData[trainNum: (trainNum + testNum), :]

    imf_list = VMD(targetData.reshape(-1, ), 6)
    imf_list = imf_list.tolist()

    # coeffs = []
    # for i in range(len(imf_list)):
    #     imf = imf_list[i]
    #     for j in range(len(imf)):
    #         part_real = imf[j].real
    #         imf[j] = part_real
    #     coeffs.append(np.array(imf).reshape(-1, 1))

    # coeffs_rest = 0
    # for i in range(len(coeffs)):
    #     coeffs_rest = coeffs_rest + coeffs[i]
    # coeffs_rest = targetData - coeffs_rest

    # coeffs[0] = coeffs[0] + coeffs[1]
    # coeffs[1] = coeffs_rest

    # # imf_list = VMD(coeffs_rest.reshape(-1, ), VMD_level)
    # # imf_list = imf_list.tolist()
    # #
    # # coeffs = []
    # # for i in range(len(imf_list)):
    # #     imf = imf_list[i]
    # #     for j in range(len(imf)):
    # #         part_real = imf[j].real
    # #         imf[j] = part_real
    # #     coeffs.append(np.array(imf).reshape(-1, 1))

    # # plt.figure(figsize=(19, 8))
    # # plt.subplot(611)
    # # plt.plot(targetData[:1000, :])
    # # plt.subplot(612)
    # # plt.plot(coeffs[0][:1000, :])
    # # plt.subplot(613)
    # # plt.plot(coeffs[1][:1000, :])
    # # plt.subplot(614)
    # # plt.plot(coeffs[2][:1000, :])
    # # plt.subplot(615)
    # # plt.plot(coeffs[3][:1000, :])
    # # plt.subplot(616)
    # # plt.plot(coeffs_rest[:1000, :])
    # # plt.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.95, hspace=0.25)
    # # plt.show()

    # ### 测试滤波效果
    # decomposed_data = []
    # for i in range(len(coeffs)):
    #     VMD_trainX, VMD_trainY, VMD_testX, VMD_testY = create_data(coeffs[i], trainNum, 4)
    #     decomposed_data.append([VMD_trainX, VMD_trainY, VMD_testX, VMD_testY])

    # print('load_data complete.\n')

    # return decomposed_data, testY
    return imf_list


filename1 = "PRSA_Data_"
filename2 = ".csv"
filename = [filename1, filename2]
# training number
startNum = 1
interval = 1
trainNum = (24 * 1000) // interval
testNum = ((24 * 100) // interval) + 4
emd_decay_rate = 1.00

dataset = read_csv_PM10(filename, trainNum, testNum, startNum, 1, interval)
VMD_list = load_data_VMD(trainNum, testNum, startNum, dataset)
np.savetxt("decomposed_data.csv", VMD_list, delimiter=",")
