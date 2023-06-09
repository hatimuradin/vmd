import math
from sklearn.metrics import mean_squared_error #均方误差       MSE
from sklearn.metrics import mean_absolute_error #平方绝对误差  MAE
from sklearn.metrics import r2_score#R square #调用
import numpy as np

# mean_squared_error(y_test,y_predict)
# mean_absolute_error(y_test,y_predict)
# r2_score(y_test,y_predict)

## array 版
# mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def MAPE(true,predict):
    L1=int(true.shape[0])
    L2=int(predict.shape[0])
    #print(L1,L2)
    if L1==L2:
        #SUM1=sum(abs(true-predict)/abs(true))
        SUM=0.0
        for i in range(L1-1):
            SUM=(abs(true[i,0]-predict[i,0])/true[i,0])+SUM
        per_SUM=SUM*100.0
        mape=per_SUM/L1
        return mape
    else:
        print("error")

#list 版
def MAPE1(true,predict):

    L1 = int(len(true))
    L2 = int(len(predict))

    if L1 == L2:

        SUM = 0.0
        for i in range(L1):
            if true[i] == 0:
                SUM = abs(predict[i]) + SUM
            else:
                SUM = abs((true[i] - predict[i]) / true[i]) + SUM
        per_SUM = SUM * 100.0
        mape = per_SUM / L1
        return mape
    else:
        print("error")


def MSE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    #mse = np.sum((predict_data-true_data)**2)/len(true_data) #跟数学公式一样的
    mse = mean_squared_error(testY[:,0], testPredict[:, 0])
    return mse


def MSE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    mse = mean_squared_error(testY[:], testPredict[:])
    return mse


def RMSE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    # rmse = mse ** 0.5
    rmse = math.sqrt( mean_squared_error(testY[:,0], testPredict[:, 0]))
    return rmse


def RMSE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    rmse = math.sqrt( mean_squared_error(testY[:], testPredict[:]))
    return rmse


def MAE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    #mae = np.sum(np.absolute(predict_data - true_data))/len(true_data)
    mae=mean_absolute_error(testY[:,0], testPredict[:, 0])
    return mae


def MAE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    mae=mean_absolute_error(testY[:], testPredict[:])
    return mae


def R2(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score


def deal_accuracy(flag1, flag2):

    rightCount = 0
    # flag1 = flag1[1:]
    for i in range(len(flag2)):
        if flag1[i] == flag2[i]:
            rightCount = rightCount+1
    accuracy = rightCount / len(flag2)

    return accuracy * 100


def deal_flag(Data, min_ramp):

    flag_temp = []

    # global minLen
    for i in range(len(Data) - 1):
        # 上升为 1.
        if (Data[i+1] - Data[i]) > min_ramp:
            flag_temp.append(1)
        # 下降为 2.
        elif (Data[i + 1] - Data[i]) < min_ramp:
            flag_temp.append(2)
        # 不变为 0.
        else:
            flag_temp.append(0)

    return flag_temp


def Evaluate_short(name, true, predict):

    rmse = round(RMSE1(true, predict), 3)
    mape = round(MAPE1(true, predict), 2)
    mae = round(MAE1(true, predict), 3)

    eva_output = '\n\nMAE_'+name+': {}'.format(mae)
    eva_output += '\nRMSE_'+name+': {}'.format(rmse)
    eva_output += '\nMAPE_'+name+': {}'.format(mape)
    result_all = [name, mae, rmse, mape]

    return eva_output, result_all


def Evaluate(name, name_short, true, predict, accuracy):

    rmse = RMSE1(true, predict)
    mape = MAPE1(true, predict)
    mae = MAE1(true, predict)
    r2 = R2(true, predict)

    eva_output = '\n\nMAE_'+name+': {}'.format(mae)
    eva_output += '\nRMSE_'+name+': {}'.format(rmse)
    eva_output += '\nMAPE_'+name+': {}'.format(mape)
    eva_output += '\nR2_'+name+': {}'.format(r2)
    eva_output += '\nACC_'+name+': {}'.format(accuracy)
    result_all = [name_short, mae, rmse, mape, r2, accuracy]

    return eva_output, result_all


def Evaluate_DL(name, name_short, true, predict, accuracy, train_time):

    rmse = RMSE1(true, predict)
    mape = MAPE1(true, predict)
    mae = MAE1(true, predict)
    r2 = R2(true, predict)

    eva_output = '\n\nMAE_'+name+': {}'.format(mae)
    eva_output += '\nRMSE_'+name+': {}'.format(rmse)
    eva_output += '\nMAPE_'+name+': {}'.format(mape)
    eva_output += '\nR2_'+name+': {}'.format(r2)
    eva_output += '\nACC_'+name+': {}'.format(accuracy)
    eva_output += '\nTIME_'+name+': {}'.format(train_time)
    result_all = [name_short, mae, rmse, mape, r2, accuracy, train_time]

    return eva_output, result_all


def perform_PE(predict, data):
    predict = predict.tolist()
    data = data.tolist()

    MAPE_list = []
    for i in range(len(predict)):
        if data[i] != 0:
            mape_value = abs(predict[i] - data[i]) / data[i]
        else:
            mape_value = abs(predict[i] - data[i])
        MAPE_list.append(mape_value)

    return (np.array(MAPE_list).reshape(-1, )) * 100


def perform_SE(predict, data):
    predict = predict.tolist()
    data = data.tolist()

    SE_list = []
    for i in range(len(predict)):
        mape_value = (predict[i] - data[i]) ** 2
        SE_list.append(mape_value)

    return np.array(SE_list).reshape(-1, )


def perform_AE(predict, data):
    predict = predict.tolist()
    data = data.tolist()

    SE_list = []
    for i in range(len(predict)):
        mape_value = abs(predict[i] - data[i])
        SE_list.append(mape_value)

    return np.array(SE_list).reshape(-1, )


def perform_MAPE_long(predict, data, length):
    mape_list = []
    for i in range(predict.shape[0] - length):
        predict_part = predict[i:i + length,]
        data_part = data[i:i + length,]
        mape_part = MAPE1(data_part, predict_part)
        mape_list.append(mape_part)
    return mape_list