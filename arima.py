# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 22:10:13 2019

@author: Yuntian Chen
"""
# Notes:
# ARMA and ARIMA are autoregressive models, and historical data of the districts to be predicted are required as training data. 
# However, the historical data of the target districts are not available in this study, and thus the autoregressive models cannot be used directly. 
# In order to fairly compare the performance of ARMA and ARIMA in the experiment, we used an extended ARMA and ARIMA in the experiment. 
# We first train an ARMA or ARIMA model in each district in the training data, and then fuse the model parameters and load the averaged model parameters to the model of the target district for prediction.

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.api import qqplot


ave_window = 1
use_diff = 0

use_ratio = 0 # dimensionless ratio
data_origin = []
data_diff = []
load_mean = []
load_std = []
data_list = ['CY','HD','FT','SJS','PG','YZ','CP',
         'MTG','FS','DX','HR','MY','SY','YQ'] # 缺少net 缺少PG，PG天气数据从2010年6月开始，缺失过多
for i in range (13):
    if use_ratio == True:
        name = 'E:/CYQ/LSTM-ENN/Grid_ratio_correctdata_correctweather/data/data_day_'+ data_list[i] +'.csv'
    else:
        name = 'E:/CYQ/LSTM-ENN/Grid_ratio_correctdata_correctweather/data/real_data_day_'+ data_list[i] +'.csv'
    #name = 'E:/Research CYT/grid/enlstm_code/ratio/data/data_day_'+ data_list[i] +'.csv'
    script = 'df = pd.read_csv(\'{0}\')'.format(name)
    script2 = 'data_origin.append(df.iloc[:,1])'
    exec (script)
    exec (script2)
    load_mean.append(data_origin[i].mean())
    load_std.append(data_origin[i].std())
    new_load = (data_origin[i]-load_mean[i])/load_std[i]
    temp1 = np.array(new_load).reshape(-1,ave_window)
    new_load = temp1.mean(1)
    data_origin[i] = pd.DataFrame(new_load)
    if use_diff == True:
        data_diff.append(data_origin[i].diff(int(24/ave_window))[int(24/ave_window):])



params_list = []
for i in range (13):
    if use_diff == True:
        script3 = 'arima_%d = ARIMA(data_diff[%d], order=(20, 0, 2)).fit()' % (i, i)
    else:
        script3 = 'arima_%d = ARIMA(data_origin[%d], order=(20, 0, 2)).fit()' % (i, i)   
    script4 = 'params_list.append(arima_%d.params)' % (i)
    exec (script3)
    exec (script4)
    print(i)
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
def forecast_transfer (params, step, model_name, endog):
    forecast_transfer = _arma_predict_out_of_sample(params, step,
                                                    model_name.resid, model_name.k_ar, 
                                                    model_name.k_ma, model_name.k_trend, 
                                                    model_name.k_exog, endog,
                                                    exog=None, method=model_name.model.method)
    pred_test = model_name.forecast(steps=int(24/ave_window))
    return forecast_transfer

def forecast_transfer_multiple_output (params, step, start, data, ID):
    N_sample = (len(data)-start)/step
    # print('fitting model...')
#     arima_temp = ARIMA(data, order=(4, 0, 4)).fit() # create the model with input data
    # print(np.shape(data))
    pred = None
    for i in range (int(N_sample)):
        data_temp = data[0:(start+i*step)]      
        endog = arima_temp.model.endog[0:(start+i*step)]
        pred_temp = forecast_transfer (params, step, arima_temp, endog)
        if pred is None:
            pred = pred_temp
        else:
            pred = np.hstack((pred, pred_temp))
        # if i%300 == 0:
        #     print('Finish the sample:', i)
    return pred
    print(i)


#####################################################################################################
a = [[5, 4, 1], [6, 8, 3], [12, 9, 7], [13, 2, 10]]
b = [[6, 8, 3, 12, 9, 7, 13, 2, 10],
     [5, 4, 1, 12, 9, 7, 13, 2, 10],
     [5, 4, 1, 6, 8, 3,  13, 2, 10],
     [5, 4, 1, 6, 8, 3, 12, 9, 7]]
for test_set_ID in range(4):
    test_ID = a[test_set_ID]
    train_ID = b[test_set_ID]
    script5 = 'arima_train_params = (params_list[%d] + params_list[%d] + params_list[%d] + \
                        params_list[%d] + params_list[%d] + params_list[%d] + \
                        params_list[%d] + params_list[%d] + params_list[%d])/9'\
                        %(train_ID[0]-1, train_ID[1]-1, train_ID[2]-1, train_ID[3]-1, train_ID[4]-1, train_ID[5]-1,\
                        train_ID[6]-1, train_ID[7]-1, train_ID[8]-1)
    exec (script5)
    # print('arima_train_params:',arima_train_params)
    print(test_ID)
    # for i in range(13):
    #     print(params_list[i])

    pred_list = []
    for j in range (len(test_ID)):
        print('Predicting test_ID:', test_ID[j])
    #     arima_train_params
        start = np.max((4,int(24/ave_window)))
        script_temp = 'arima_temp = arima_%d' % ((test_ID[j]-1))   
        exec (script_temp) 
        print(arima_temp)
        if use_diff == True:
            pred = forecast_transfer_multiple_output (arima_train_params, int(24/ave_window), start, (data_diff[test_ID[j]-1]), (test_ID[j]-1))
        else:
            pred = forecast_transfer_multiple_output (arima_train_params, int(24/ave_window), start, (data_origin[test_ID[j]-1]), (test_ID[j]-1))
        pred_list.append(pred)    
        
    for j in range (len(test_ID)):
        pred_real = pred_list[j]
        if use_diff == True:
            data_real = np.array(data_diff[test_ID[j]-1][-len(pred_real):])
            plt.figure(figsize=(100, 20))
    #         plt.plot(np.cumsum(np.array(pred_real)), label='prediction', color='red', alpha=0.4)
    #         plt.plot(np.cumsum(np.array(data_real)), label='target', color='black', alpha=0.4)
        else:        
            data_real = np.array(data_origin[test_ID[j]-1][-len(pred_real):])
        print('ID:',test_ID[j])
        print(np.shape(pred_real))
        print(np.shape(data_real))
        plt.figure(figsize=(100, 20))
        # plt.plot(pred_real, label='prediction', color='red', alpha=0.4)
        # plt.plot(data_real, label='target', color='black', alpha=0.4)
        
        error = []
        for i in range(len(data_real)):
            error.append(data_real[i] - pred_real[i]) 
        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)#target-prediction之差平方 
            absError.append(abs(val))#误差绝对值
        print("MSE = ", sum(squaredError) / len(squaredError))#均方误差MSE



for i in range (13):
    script_save = '''arima_%d.save('arima_20_0_2_id%d.pkl')''' % (i, i)   
    exec (script_save)
