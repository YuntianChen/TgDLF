# -*- coding: utf-8 -*-
"""
Created on Wed May 23 19:15:00 2018

@author: Yuntian Chen
"""

import torch
import numpy as np
import random


class DefaultConfiguration:
    def __init__(self):
        # 实验参数设置
        self.seed = 666
        self.supplement = 'test_'
        self.experiment_ID = '6101'
        self.test_set_ID = 1
        self.deviceID = 0
        #self.deviceID = self.test_set_ID % 2
        self.epoch = 5

        
        self.ERROR_PER = 0.02
        self.path = 'E' + self.experiment_ID

        self.GAMMA = 10
        self.drop_last = False
        # 数据集相关参数
        self.test_pro = 0.3
        self.total_well_num = 14
        self.train_len = int(24*4)  # 训练数据长度为96,对应4天
        self.predict_len = int(24 * 1)
        # 神经网络设置
        self.T = 1
        self.ne = 100
        self.use_annual = 1
        self.use_quarterly = 0
        self.use_monthly = 0      
        self.input_dim = 9  # annual:9; quarterly:12; monthly:20
        self.hid_dim = 30  # 隐含节点个数
        self.num_layer = 1  # RNN层数
        self.drop_out = 0.3
        self.output_dim = 1
        # 训练设置
        self.batch_size = 512
        self.num_workers = 0
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        #self.max_epoch = 30
        self.display_step = 10

        # 后续内容不包含需要修改的变量
        # 生成训练集与测试集
        self.reverse = False   # 是否选择反向预测,True表示反向,False表示正向
       
        # cross validation experiment
        self.test_well_num = int(self.test_pro*self.total_well_num)
        self.train_well_num = self.total_well_num - self.test_well_num  # 训练数据中井的数目
        self.rdn_lst = np.arange(1, self.total_well_num+1, 1)
        random.shuffle(self.rdn_lst)
        a = [[5, 4, 1], [6, 8, 3], [12, 9, 7], [13, 2, 10]]
        b = [[6, 8, 3, 12, 9, 7, 13, 2, 10],
             [5, 4, 1, 12, 9, 7, 13, 2, 10],
             [5, 4, 1, 6, 8, 3,  13, 2, 10],
             [5, 4, 1, 6, 8, 3, 12, 9, 7]]
        self.test_ID = a[self.test_set_ID]
        self.train_ID = b[self.test_set_ID]
        #self.test_ID = np.array(self.rdn_lst[0:self.test_well_num])  # 一个list包含所有test数据
        #self.train_ID = np.array(self.rdn_lst[self.test_well_num:self.total_well_num])
        self.data_list = ['CY', 'HD', 'FT', 'SJS', 'PG', 'YZ', 'CP',
                          'MTG', 'FS', 'DX', 'HR', 'MY', 'SY', 'YQ']
        #self.test_ID = [9, 5, 12, 2]

                
        # 训练设置

        self.info = "e{}-{}-{}-{}-{}-{}".format(self.experiment_ID, self.path, str(self.hid_dim), str(self.batch_size),
                                                str(self.ne), str(self.ERROR_PER))
        
        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False    
        # 保存加载、预测设置
        self.print_freq = 10
        self.checkpoint = 'grid_'+self.supplement+self.experiment_ID+'.pkl'
        
    def update(self):
        
        random.shuffle(self.rdn_lst)
        #self.test_ID = np.array(self.rdn_lst[0:self.test_well_num])  # 一个list包含所有test数据
        #self.train_ID = np.array(self.rdn_lst[self.test_well_num:self.total_well_num])
        self.experiment_ID = str(int(self.experiment_ID) + 1)
        self.checkpoint = 'well'+self.supplement+'{:d}.pkl'.format(int(self.experiment_ID))        


config = DefaultConfiguration()
