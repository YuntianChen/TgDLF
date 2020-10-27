import numpy as np
from configuration import config
import torch.utils.data
import pandas as pd
from sklearn import preprocessing
import copy


# read file and change the head name
def read_file(path):
    df = pd.read_csv(path)
    df.columns = config.head
    return df


# make dataset using moving window with the step of -> window_step
def make_dataset(data, window_size):
    i = 0
    while i + window_size - 1 < len(data):
        yield data[i:i+window_size]
        i += config.window_step


def normalize(x):
    scaler = preprocessing.StandardScaler().fit(x)
    return scaler.transform(x), scaler


class TextDataset(torch.utils.data.Dataset):
    scaler = None
    dataset_scaler = {}
    test_data = {}

    def __init__(self):
        self.df_list = []
        for i in range(config.well_num):
            # the test and train data will be the unnormalized data
            filename = config.data_prefix.format(i+1)
            df = read_file(filename)
            df['DEPT'] = np.arange(1, len(df)+1)
            self.df_list.append(df)
            if i+1 in config.test_ID:
                self.test_data[i+1] = df[config.columns_target]
        self.dataset = pd.concat(self.df_list, axis=0, ignore_index=True)
        for feature in config.columns_target:
            self.dataset_scaler[feature] = preprocessing.StandardScaler().fit(self.dataset[feature].values.reshape(-1, 1))
        self.input_data, self.target_data = self.train_dataset()
        self.line_num = len(self.input_data)

    def reset_train_dataset(self):
        self.input_data, self.target_data = self.train_dataset()
        self.line_num = len(self.input_data)

    def reset_test_dataset(self):
        for items in config.test_ID:
            self.df_list[items-1][config.columns_target] = self.test_data[items][config.columns_target].values

    def train_dataset(self):
        input_data = []
        target_data = []
        selected = config.columns[:config.input_dim+config.output_dim]
        for items in config.train_ID:
            data = copy.copy(self.df_list[items-1])
            input_ = np.array(list(make_dataset(
                normalize(data[selected[:config.input_dim]].values)[0], config.train_len)))
            target_ = np.array(list(make_dataset(
                self.dataset_scaler[selected[-1]].transform(data[selected[-1]].values.reshape(-1, 1)), config.train_len)))
            input_data.append(input_)
            target_data.append(target_)
        return np.concatenate(input_data), np.concatenate(target_data)

    def test_dataset(self, index):
        selected = config.columns[:config.input_dim+1]
        data = copy.copy(self.df_list[index-1])
        input_ = normalize(data[selected[:config.input_dim]].values)[0]
        target_ = self.dataset_scaler[selected[-1]].transform(data[selected[-1]].values.reshape(-1, 1))
        self.scaler = self.dataset_scaler[selected[-1]]
        return input_, target_

    def inverse_normalize(self, x):
        return self.scaler.inverse_transform(x)

    def __getitem__(self, index):
        return self.input_data[index], self.target_data[index]
    
    def __len__(self):
        return self.line_num
