# -*- coding: utf-8 -*-
import torch


class DefaultConfiguration:

    def __init__(self):
        # 数据相关设置
        self.well_num = 6
        self.train_len = 200
        self.shrink_len = 10
        self.window_step = 100
        self.head = ['DEPT', 'RMG', 'RMN', 'RMN-RMG', 'CAL', 'SP', 'GR', 'HAC', 'BHC', 'DEN']
        self.columns = ['DEPT', 'RMN-RMG', 'CAL', 'SP', 'GR', 'HAC', 'BHC', 'DEN']
        self.columns_input = ['DEPT', 'RMN-RMG', 'CAL', 'SP', 'GR']
        self.columns_target = ['HAC', 'BHC', 'DEN']
        # 神经网络设置
        self.ERROR_PER = 0.02 # 0.002
        self.drop_last = False
        self.input_dim = 5
        self.hid_dim = 30
        self.num_layer = 1
        self.drop_out = 0.3
        self.output_dim = 1
        # 训练设置
        # training parameters
        self.ne = 100
        self.T = 1
        self.batch_size = 32
        self.num_workers = 1
        self.epoch = 3
        self.GAMMA = 10
        # 实验参数设置
        self.train_ID = [1, 2, 3, 4, 5]
        self.test_ID = [6, ]
        self.data_prefix = 'data/vertical_all_A{}.csv'
        self.well_ID = 'A4'
        self.experiment_ID = '032' \
                             ''
        self.deviceID = 1

        self.path = 'Experiments/{}'.format(self.experiment_ID)
        self.info = "Exp{}-{}-{}-{}-{}".format(self.experiment_ID, str(self.hid_dim),
                                               str(self.batch_size), str(self.ne), str(self.ERROR_PER))
        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.text_path = 'data/vertical_all_{}.xls'.format(self.well_ID)


config = DefaultConfiguration()
