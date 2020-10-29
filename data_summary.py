# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 22:10:13 2019

@author: Yuntian Chen
"""

import numpy as np
from grid_configuration import config
import torch.utils.data
import torch
import pandas as pd
# hyper parameters
use_forcast_weather = 1
delta_weather = 0

use_ratio = 0 # dimensionless ratio
use_mean_ratio = 0
use_different_mean_ratio = 0 # 各个区用不同的ratio
use_CV_ratio = 0

use_weather_error_test = 0 # 此时只有test有误差
weather_error_test = 0.60
use_weather_error_train = 0 # 此时只有train有误差
weather_error_train = 0.05

def ave_ratio (data_origin):    
    # calculate the average ratio of all the districts
    # inputs: the original data of all districts
    # outputs:  mean_ratio_all is a (num_districts*num_data) matrix. it is the mean_ratio of all districts
    #           mean_ratio_all_ave is an array with size of (num_data,). it is the average mean_ratio over all districts
    mean_ratio_all = None
    for i in range (14):
        load = data_origin[i]
        load_raw_array = load.iloc[:, 1]  # 32616个ratio
        input_load = np.array(load_raw_array)
        data_num = np.shape(input_load)[0]
        week_num = int(data_num/168) # calculate the number of weeks
        # reshape loads to (num of hours in one week) * (num of weeks)
        delet_ID = np.arange(week_num*168, data_num)
        input_load_del = np.delete( input_load, delet_ID, 0)# 产生整数周的结果
        input_load_week = input_load_del.reshape(168, week_num) # 168(num of hours in one week) * num of weeks
        # calculate the average ratio in one week
        input_load_week_mean = np.mean(input_load_week, axis=1)
        # generate the average ratio for the length of data_num
        mean_ratio = None
        for i in range (week_num+1):
            if mean_ratio is None:
                mean_ratio = input_load_week_mean
            else:
                mean_ratio = np.hstack((mean_ratio, input_load_week_mean))
        delet_ID = np.arange(data_num, np.shape(mean_ratio)[0])
        mean_ratio = np.delete( mean_ratio, delet_ID, 0).reshape(1,-1)
        # save the mean_ratio of all districts
        if mean_ratio_all is None: # mean_ratio_all is the mean_ratio of all the districts
            mean_ratio_all = mean_ratio
        else:
            mean_ratio_all = np.vstack((mean_ratio_all, mean_ratio))
    mean_ratio_all_ave = np.mean(mean_ratio_all, axis=0) # mean_ratio_all_ave is the average of mean_ratio_all over 14 districts
    #np.savetxt('load_results.csv', np.array(mean_ratio_all).T, delimiter=',')
    mean_ratio_group1 = (np.sum(mean_ratio_all, axis=0)-mean_ratio_all[4,:]-mean_ratio_all[3,:]-mean_ratio_all[0,:])/9
    mean_ratio_group2 = (np.sum(mean_ratio_all, axis=0)-mean_ratio_all[5,:]-mean_ratio_all[7,:]-mean_ratio_all[2,:])/9
    mean_ratio_group3 = (np.sum(mean_ratio_all, axis=0)-mean_ratio_all[11,:]-mean_ratio_all[8,:]-mean_ratio_all[6,:])/9
    mean_ratio_group4 = (np.sum(mean_ratio_all, axis=0)-mean_ratio_all[12,:]-mean_ratio_all[1,:]-mean_ratio_all[9,:])/9
    mean_ratio_group = np.vstack(( np.vstack(( np.vstack(( mean_ratio_group1,mean_ratio_group2 )),mean_ratio_group3 )),mean_ratio_group4 ))
    #np.savetxt('ratio_group.csv', np.array(mean_ratio_group).T, delimiter=',')
    return mean_ratio_all, mean_ratio_all_ave, mean_ratio_group # 都是32616长度的。但是在真正预测的时候，对于output，缺少第一天，应该是32592.因此对于预测要删除开头24个

# load data
data_origin = []
data_list = ['CY','HD','FT','SJS','PG','YZ','CP',
         'MTG','FS','DX','HR','MY','SY','YQ'] # 缺少net 缺少PG，PG天气数据从2010年6月开始，缺失过多
for i in range (14):
    if use_ratio == True:
        name = 'E:/CYQ/LSTM-ENN/Grid_ratio_correctdata_correctweather/data/data_day_'+ data_list[i] +'.csv'
    else:
        name = 'E:/CYQ/LSTM-ENN/Grid_ratio_correctdata_correctweather/data/real_data_day_'+ data_list[i] +'.csv'
    #name = 'E:/Research CYT/grid/enlstm_code/ratio/data/data_day_'+ data_list[i] +'.csv'
    script = 'df = pd.read_csv(\'{0}\')'.format(name)
    script2 = 'data_origin.append(df)'
    exec (script)
    exec (script2)
    #print(i)
print('shape of df',np.shape(np.array(df)))
# 根据各区数据对load分别归一化,生产load_normal
load_mean = np.zeros((14, 4)) # 3对应的是load+weather+风速
load_mean_save = np.zeros((14, 1))
load_std = np.zeros((14, 4))
load_std_save = np.zeros((14, 1))
load_normal = [] # 对应正则化后的load和weather，维度为x*1，已经拼接成一列了
# generate the mean_ratio_all
if use_mean_ratio == True:
    mean_ratio_all, mean_ratio_all_ave, mean_ratio_group = ave_ratio (data_origin)
    print ('shape of mean_ratio_all is:', np.shape(mean_ratio_all))
    print ('shape of mean_ratio_all_ave is:', np.shape(mean_ratio_all_ave))

# 生成基本数据  generate the whole dataset and normalize a part of it
# generate the load_normal, include load, weather(tem, rhu, wind)    
for i in range (14):
    load = data_origin[i]
    load_raw_array = load.iloc[:, 1:2]  
    if use_mean_ratio == True:
        if use_different_mean_ratio == True:
            load_raw_array = load_raw_array - mean_ratio_all[i,:].reshape(-1,1)
        elif use_CV_ratio == True:
            load_raw_array = load_raw_array - mean_ratio_group[config.test_set_ID,:].reshape(-1,1)
        else:
            load_raw_array = load_raw_array - mean_ratio_all_ave.reshape(-1,1)
    weather_raw_array = load.iloc[:, 2:5]
    # calculate the change of weather
    if delta_weather == True:
        weather_raw_array = np.array(weather_raw_array)
        weather_raw_array_pre = np.concatenate((weather_raw_array[0].reshape(1,-1), weather_raw_array[0:-1]), axis=0)
        weather_raw_array = weather_raw_array - weather_raw_array_pre
    # the load and weather
    load_raw_array = np.concatenate((load_raw_array, weather_raw_array), axis=1)
    # normalization
    load_mean[i] = load_raw_array.mean(axis=0)
    load_std[i] = load_raw_array.std(axis=0)
    #load_mean_save[i] = load_mean[i, 0]
    #load_std_save[i] = load_mean[i, 0]
    if use_ratio == True:   
        load_mean[i,0]=0
        load_std[i,0]=1
    load_normal_array = load_raw_array 
    load_normal.append(load_normal_array)  
# generate the whole dataset (load, weather, rain, Sat, Mon, Date, Weekend)
data_normal = []
for i in range (14):
    #weather = np.array(data_origin[i].iloc[:,2:4]) #weather 已经在load中一起归一化了
    rain = data_origin[i].iloc[:,5:6] # 降雨量
    if delta_weather == True:
        rain = np.array(rain)
        rain_pre = np.concatenate((rain[0].reshape(1,-1), rain[0:-1]), axis=0)
        rain = rain - rain_pre
        #print ('rain shape:', np.shape(rain))
    date = data_origin[i].iloc[:,7:19]
    weekend = data_origin[i].iloc[:,6:7]
    Sat = data_origin[i].iloc[:,19:20]
    Mon = data_origin[i].iloc[:,20:21]
    data_normal_array = np.concatenate((load_normal[i], rain, Sat, Mon,   date, weekend), axis=1)
    #load_normal 0,1,2,3, rain 4, Sat 5, Mon 6,   date 7:18, weekend 19
    data_normal.append(data_normal_array)



train_data_input = None
train_data_output = None
record = np.zeros((config.train_well_num,3))
train_data_input_all_list = [] # 用来记录训练数据的列表，并没有根据config.train_len提取数据。并不是用来训练LSTM的
train_data_output_all_list = [] # 用来记录训练数据的列表，并没有根据config.train_len提取数据。并不是用来训练LSTM的

# 数据预处理，sliding windows结果
data_input_list = []
data_output_list = []
for k in range(14):
    data = data_normal[k] 
    test_num = data.shape[0] # 获取表的行数
    # date和weekend可以提前获知，所以滑动到当天对应的值
    next_date = np.zeros((test_num,12))
    next_weekend = np.zeros((test_num,1))
    next_Sat = np.zeros((test_num, 1))
    next_Mon = np.zeros((test_num,1))
    if use_forcast_weather == True:
        next_weather = np.zeros((test_num,4))
    data_output = np.zeros((test_num-24,1))
    for i in range (test_num-24):
        next_Sat[i] = data[i + 24, 5]
        next_Mon[i] = data[i + 24, 6]
        next_date[i] = data[i + 24, 7:19]
        next_weekend[i] = data[i + 24, 19]
        if use_forcast_weather == True:
            next_weather[i] = data[i + 24, 1:5]
    for i in range (24):
        next_Sat[test_num-24+i] = -999 # 数据向前滑动后最终部分缺失，填充-999
        next_Mon[test_num-24+i] = -999 # 数据向前滑动后最终部分缺失，填充-999
        next_date[test_num - 24 + i] = -999  # 数据向前滑动后最终部分缺失，填充-999
        next_weekend[test_num - 24 + i] = -999  # 数据向前滑动后最终部分缺失，填充-999
        if use_forcast_weather == True:
            next_weather[test_num - 24 + i] = -999
    # 选择时间尺度
    if config.use_annual:
        date_info = next_date[:,5:6] + next_date[:,6:7] + next_date[:,7:8] + next_date[:,8:9]
    if config.use_monthly:
        date_info = next_date
    if config.use_quarterly:
        first = next_date[:,0:1] + next_date[:,1:2] + next_date[:,2:3]
        second = next_date[:,3:4] + next_date[:,4:5] + next_date[:,5:6]
        third = next_date[:,6:7] + next_date[:,7:8] + next_date[:,8:9]
        fourth = next_date[:,9:10] + next_date[:,10:11] + next_date[:,11:12]
        date_info = np.hstack((np.hstack((np.hstack((first, second)),third)),fourth))

# generate the whole dataset
    if use_forcast_weather == True:
        data_input_all = np.concatenate((data, next_weather, next_Sat, next_Mon, date_info, next_weekend),axis=1)#data包含的内容：load_normal[i] 0,1,2,3, rain 4, Sat 5, Mon 6,   date 7:19, weekend 19
        #data结构：load_normal 0,1,2,3, rain 4, Sat 5, Mon 6,   date 7:18, weekend 19
    else:
        data_input_all = np.concatenate((data, next_Sat, next_Mon, date_info, next_weekend),axis=1)           
    # 生成data_input
    delet_ID = np.arange(test_num-24,test_num)
    data_input = np.delete( data_input_all,delet_ID,0) # 删除掉最后缺少数据的一天（96行）
    if use_forcast_weather == True:
        data_input = np.delete( data_input,[1,2,3,4,   5,6,   7,8,9,10,11,12,13,14,15,16,17,18,   19],1)
        #data结构：load_normal 0,1,2,3, rain 4, Sat 5, Mon 6,   date 7:18, weekend 19
        # 1,2,3,4对应weather，4为rain。5,6为sat&mon。7-18为date， 19为weekend
    else:
        data_input = np.delete( data_input,[5,6,   7,8,9,10,11,12,13,14,15,16,17,18,   19],1) # 删除掉当前的date和weekend，用下一天的（对应output的）。删除之前存在两组date和weekend
    # 9 inputs: load, T, Rhu, Wind, Rain, Sat, Mon, date_info, Week
    data_input_list.append(data_input)
    # 生成data_output
    for i in range (test_num-24):
        data_output[i] = data[i+24,0] # 基于load_raw_array计算的，因此在use_mean_ratio == True时，已经是去除mean_ratio
    data_output_list.append(data_output)
            
            
print(np.shape(data_input_list))            
inputdata_stat_all = []
for i in range (14):
    inputdata = data_input_list[i]
    inputdata_mean = inputdata.mean(axis=0).reshape(1,-1)
    inputdata_max = inputdata.max(axis=0).reshape(1,-1)
    inputdata_min = inputdata.min(axis=0).reshape(1,-1)
    inputdata_meidan = np.median(inputdata,axis=0).reshape(1,-1)
    inputdata_std = inputdata.std(axis=0).reshape(1,-1)
    inputdata_stat = np.hstack( (np.hstack( (np.hstack( (np.hstack((inputdata_mean, inputdata_max)), inputdata_min)), inputdata_meidan) ), inputdata_std ))
    #print(np.shape(inputdata_stat))
    inputdata_stat_all.append(inputdata_stat)
inputdata_stat_all = np.array(inputdata_stat_all).reshape(14,-1)
np.savetxt('data_summary.csv', inputdata_stat_all, delimiter=',')

