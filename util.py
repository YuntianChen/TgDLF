import numpy as np
import pickle
import os


def list_to_csv(li):
    result = ''
    for i in li:
        result += '{},'.format(i)
    return result


def Regeneralize (result, mean, std): # 输入的result是个array
    #row_size = result.shape[0]
    mean_matrix = mean
    std_matrix = std
    #mean_matrix = np.tile(mean_matrix, (row_size, 1))
    #std_matrix = np.tile(std_matrix, (row_size, 1))
    result_real_scale = (result * std_matrix) + mean_matrix
    return result_real_scale


def get_file_list(prefix, folder):
    result = []
    file_list = os.listdir(folder)
    for file in file_list:
        if file.endswith(prefix):
            result.append(os.path.join(folder, file))
    return sorted(result)


def save_var(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def shrink(data, size):
    i = 0
    while i+size < data.shape[1]:
        #print(type(data))
        yield data[:, i:i+size, :].mean(dim=1)
        i += size


class Record:
    
    def __init__(self):
        self.data = []
        self.mean = []
        self.count = 0
        self.reset()

    def reset(self):
        self.data = []
        self.mean = []
        self.count = 0

    def update(self, val, n=1):
        self.data.append(val)
        if type(val) == list:
            self.mean.append(np.mean(val))
        else:
            self.mean.append(val)
        self.count += n
    
    def get_latest(self, mean=True):
        if mean:
            return self.mean[-1]
        else:
            return self.data[-1]

    def delta(self):
        return abs(self.mean[-1] - self.mean[-2])

    def bigger(self):
        return self.mean[-1] - self.mean[-2] > 0

    def check(self, val):
        return (self.mean[-1] - np.mean(val)) < 0
