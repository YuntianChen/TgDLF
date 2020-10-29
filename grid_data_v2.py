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
from scipy import signal
# hyper parameters
use_forcast_weather = 0
delta_weather = 0
use_filter = 1 # 对获得的Dimensionless trend进行低通滤波

use_ratio = 1 # dimensionless ratio
use_mean_ratio = 1
use_different_mean_ratio = 0 # 各个区用不同的ratio
use_CV_ratio = 1 # 使用交叉检验中的ratio

use_weather_error_test = 0 # 此时只有test有误差
weather_error_test = 0.60
use_weather_error_train = 0 # 此时只有train有误差
weather_error_train = 0.05

def ave_ratio (data_origin, use_filter):    
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
        print('original:',np.mean(input_load_week_mean))
        #print('original:',np.max(input_load_week_mean)-np.min(input_load_week_mean))
        if use_filter == True:
            # 低通滤波
            b, a = signal.butter(8, 0.2, 'lowpass')
            filter_input_load_week_mean = signal.filtfilt(b, a, input_load_week_mean)
            # 放缩到过滤前ratio的尺度。因为过滤会使得ratio的尺度降低。
            filter_input_load_week_mean = (filter_input_load_week_mean-np.min(filter_input_load_week_mean)) / (np.max(filter_input_load_week_mean)-np.min(filter_input_load_week_mean))
            input_load_week_mean = filter_input_load_week_mean * (np.max(input_load_week_mean)-np.min(input_load_week_mean)) + np.min(input_load_week_mean)
            print('filtered:',np.mean(input_load_week_mean))
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

    mean_ratio_all_ave = np.mean(np.delete(mean_ratio_all,[10,13],0), axis=0) # mean_ratio_all_ave is the average of mean_ratio_all over 14 districts
    #np.savetxt('load_results.csv', np.array(mean_ratio_all).T, delimiter=',')
    mean_ratio_group1 = (np.sum(np.delete(mean_ratio_all,[10,13],0), axis=0)-mean_ratio_all[4,:]-mean_ratio_all[3,:]-mean_ratio_all[0,:])/9
    print('the sum of filtered:',np.mean(np.sum(np.delete(mean_ratio_all,[10,13],0), axis=0)))
    mean_ratio_group2 = (np.sum(np.delete(mean_ratio_all,[10,13],0), axis=0)-mean_ratio_all[5,:]-mean_ratio_all[7,:]-mean_ratio_all[2,:])/9
    mean_ratio_group3 = (np.sum(np.delete(mean_ratio_all,[10,13],0), axis=0)-mean_ratio_all[11,:]-mean_ratio_all[8,:]-mean_ratio_all[6,:])/9
    mean_ratio_group4 = (np.sum(np.delete(mean_ratio_all,[10,13],0), axis=0)-mean_ratio_all[12,:]-mean_ratio_all[1,:]-mean_ratio_all[9,:])/9
    mean_ratio_group = np.vstack(( np.vstack(( np.vstack(( mean_ratio_group1,mean_ratio_group2 )),mean_ratio_group3 )),mean_ratio_group4 ))
    print('the group of filtered:',np.mean(mean_ratio_group, axis=1))
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
    mean_ratio_all, mean_ratio_all_ave, mean_ratio_group = ave_ratio (data_origin, use_filter)
    print ('shape of mean_ratio_all is:', np.shape(mean_ratio_all))
    print ('shape of mean_ratio_all_ave is:', np.shape(mean_ratio_all_ave))
    #np.savetxt('test.csv',  mean_ratio_group.T, delimiter=',')  


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
    load_normal_array = (load_raw_array - load_mean[i])/load_std[i]
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

'''
# 对比低通滤波前后的dimentionless trend
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
mean_ratio_all, mean_ratio_all_ave, mean_ratio_group_filter = ave_ratio (data_origin, True)
mean_ratio_all, mean_ratio_all_ave, mean_ratio_group = ave_ratio (data_origin, False)
plt.figure(figsize = (10,5),dpi=300)
plt.plot(np.arange(0,168),  mean_ratio_group_filter[3][0:168])
plt.plot(np.arange(0,168),  mean_ratio_group[3][0:168])
plt.gca().xaxis.set_major_locator(MultipleLocator(24))
#plt.gca().yaxis.set_major_locator(MultipleLocator(0.02))
#plt.xlim(0,168)
plt.xlabel(u"Hour (h)")
plt.ylabel(u'Dimensionless trend')
plt.show()
'''

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,config_instance): 
        config = config_instance
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
        # 调整输出值对应的mean_ratio_all_ave。因为对于output而言，缺少第一天
        mean_ratio_all_pred = np.zeros((14, test_num-24))
        mean_ratio_all_ave_pred = np.zeros((test_num-24,1))
        mean_ratio_group_pred = np.zeros((4, test_num-24))
        for i in range (test_num-24):
            if use_mean_ratio == True:
                if use_different_mean_ratio == True:
                    for j in range (14):
                        mean_ratio_all_pred[j,i] = mean_ratio_all[j,i+24]
                elif use_CV_ratio == True:
                    for j in range (4):
                        mean_ratio_group_pred[j,i] = mean_ratio_group[j,i+24]
                else:
                    mean_ratio_all_ave_pred[i] = mean_ratio_all_ave[i+24]

    # test_data_input 和 test_data_output 是为了对test数据进行预测            
        test_data_input_list = []
        test_data_output_list = []
        test_num_list = []
        for k in (config.test_ID):# 测试数据中包含的井的ID
            test_data_input = data_input_list[k-1]
            if use_weather_error_test == True:
                    test_data_input[:, 1:5] = test_data_input[:, 1:5]*(1+np.random.randn()*weather_error_test)           
            test_data_output = data_output_list[k-1]
            test_data_input_list.append(test_data_input)
            test_data_output_list.append(test_data_output)
            test_num = test_data_input.shape[0]  # 获取表的行数,-1获得训练数据个数
            test_num_list.append(test_num)
    # 针对train data进行分析
        No_train = 0
        for k in (config.train_ID):
            No_train = No_train+1 # 第几个train
            # 提取第k个train的数据
            table_input = data_input_list [k-1] # 打开k的数据
            table_output = data_output_list [k-1] # 打开k的数据
            # 计算训练数据长度
            len_num = int((data_origin[k-1].shape[0]/24)-(config.train_len/24))+1 -1 # 对应k的天数-1, 最后一个-1是因为outputdata滑动窗口导致少了一天
            print('len_num:')
            print(len_num)
            # 生成训练数据集的输入与输出
    # train_data_input 和 train_data_output 都是为了训练神经网络的，所以是3维 
            # 生成train_data_input
            train_data_input_temp = np.zeros((len_num, config.train_len, config.input_dim)) # len_num为总共训练数据样本数目，train_len是每个样本的序列长度，
            print('test1, np.shape(table_input):', np.shape(table_input))
            print('test2, len_num:', len_num)
            for i in range (len_num):
                for j in range (config.input_dim):
                    train_data_input_temp[i,:,j] = table_input[:,j][(i*24):((i*24)+config.train_len)]
                if use_weather_error_train == True:
                    train_data_input_temp[i,:,1:5] = train_data_input_temp[i,:,1:5]*(1+np.random.randn()*weather_error_train)# add noise to the training data  
            if train_data_input is None:
                train_data_input = train_data_input_temp
            else:
                train_data_input = np.concatenate((train_data_input, train_data_input_temp), axis = 0)
            # 记录
            record[No_train-1,0] = np.shape(train_data_input)[0]
            record[No_train-1,1] = np.shape(train_data_input_temp)[0]
            record[No_train-1,2] = k
            # 生成train_data_output
            train_data_output_temp = np.zeros((len_num, config.train_len, config.output_dim)) 
            for i in range (len_num):
                for j in range (config.output_dim):
                    train_data_output_temp[i,:,j] = table_output[:,j][(i*24):((i*24)+config.train_len)]
            if train_data_output is None:
                train_data_output = train_data_output_temp
            else:
                train_data_output = np.concatenate((train_data_output, train_data_output_temp), axis = 0) 
    # train_data_input_all_list 和 train_data_output_all_list 是为了对全体train数据进行预测，所以是2维。
            train_data_input_all_list.append (table_input)
            train_data_output_all_list.append (table_output)


    # 输出数据
        self.line_num = np.shape(train_data_input)[0] # LSTM训练数据的样本数目
    # test_data_input 和 test_data_output 是为了对test数据进行预测        
        self.test_data_input_list = test_data_input_list
        self.test_data_output_list = test_data_output_list
        self.test_num_list = test_num_list
        self.mean_ratio_all_ave_pred = mean_ratio_all_ave_pred
        self.mean_ratio_all_pred = mean_ratio_all_pred
        self.mean_ratio_group_pred = mean_ratio_group_pred
    # train_data_input 和 train_data_output 都是为了训练神经网络的，所以是3维.由于不同样本来源进行了合并，所以不为list         
        self.train_data_input = train_data_input
        self.train_data_output = train_data_output
        self.record = record
    # train_data_input_all_list 和 train_data_output_all_list 是为了对全体train数据进行预测，所以是2维。因为一共5口井，所以先建立一个list，分别保存        
        self.train_data_input_all_list = train_data_input_all_list
        self.train_data_output_all_list = train_data_output_all_list
        
    def __getitem__(self, index):
        return self.train_data_input[index], self.train_data_output[index]
    
    def __len__(self):
        return self.line_num