# TgDLF
Theory-guided deep-learning load forecasting is a short-term load forecasting model that combines domain knowledge and machine learning algorithms. (see [the paper of TgDLF](https://www.enerarxiv.org/page/thesis.html?id=2022))

The model can predict future load based on historical load, weather forecast data, and the calendar effect. Specifically, the grid load is first converted into a load ratio to avoid the impact of different data distributions in different time periods in the same district. The load ratio is then decomposed into dimensionless trends that can be calculated in advance based on domain knowledge, and local fluctuations that are estimated by EnLSTM (please refer to [the code of EnLSTM](https://github.com/YuntianChen/EnLSTM) and [the paper of EnLSTM](https://arxiv.org/ftp/arxiv/papers/2004/2004.13562.pdf)). EnLSTM combines the advantages of ensemble neural network (ENN) and long short-term memory (LSTM), that is, it has the advantages of small data learning (few-shot learning), and can handle time series data problems, which are desirable for load forecasting problems.

If you encounter any problems in using the code, please contact Yuntian Chen: cyt_cn@126.com.


# The guide for TgDLF:
## Data Preperation and Model Definition

### Dataset Decription

The dataset used in this example contains data files formated as .csv files with header of feature names.
The original data file of this study is a 32616*21 matrix. The 32616 sample points correspond to hourly observations of 1359 days (1359*24=32616). The 21 features are:
Time, e.g. 1/2/2008  00:00:00; 
Load ratio, e.g. 0.9294; 
Temperature, e.g. -1.8; 
Humidity, e.g. 26; 
Wind speed, e.g. 1.8; 
Precipitation, e.g. 0; 
Whether it is weekend, e.g. 0; 
Whether it is Jan./Feb./…/Nov./Dec. (a 12-dimensional one-hot-code); 
Whether it is Saturday, e.g. 0; 
Whether it is Monday, e.g. 0.

The “grid_data_v2.py” shows the data processing details. And the line 229 of the code file “grid_data_v2.py” shows the final processed dataset, which is depends on the experiment setting (such as whether use the forecast data).

It should be noted that although various features are involved in TgDLF, it is theoretically compatible with any numerical feature. Users can select features based on actual conditions (such as the availability of data). Of course, different feature combinations will affect the accuracy of the model. 

### Loading dataset

 ```python
 text = TextDataset()
```

### Experiment settings for dataset
These settings are used for the experiments in [the paper of TgDLF](https://www.enerarxiv.org/page/thesis.html?id=2022)
These parameters do not need to be changed in the applications, and the default values should be OK.
```python
use_forcast_weather = 0 # whether to use the forecast data, 1 means True, 0 means False
use_filter = 1 # low pass filter for the dimensionless trend

use_ratio = 1 # whether to use dimensionless ratio
use_mean_ratio = 1 # Whether to use the averaged ratio as the dimensionless ratio (this is only a naive method to determine the dimensionless trend)
use_different_mean_ratio = 0 # whether to use different ratio for different districts (this indicates that each district only uses its own dimensionless trend)
use_CV_ratio = 1 # whether to use different ratio for different groups of districts (this indicates that different groups have different dimensionless trends, and the districts within each group use the same dimensionless trend)

use_weather_error_test = 0 # adding error to the test dataset
weather_error_test = 0.60 # the scale of error
use_weather_error_train = 0 # adding error to the training dataset
weather_error_train = 0.05 # the scale of error
```

---
### Neural Network Definition

```python
class netLSTM_full(nn.Module):
    def __init__(self):
        super(netLSTM_full, self).__init__()
        self.lstm = nn.LSTM(config.input_dim, config.hid_dim,
                            config.num_layer, batch_first=True, dropout=config.drop_out)
        self.fc2 = nn.Linear(config.hid_dim, int(config.hid_dim/2))
        self.fc3 = nn.Linear(int(config.hid_dim/2), config.output_dim)
        # self.fc4 = nn.Linear(int(config.hid_dim/2), int(config.hid_dim/2))
        self.bn = nn.BatchNorm1d(int(config.hid_dim / 2))
    def forward(self, x, hs=None, use_gpu=config.use_gpu):
        batch_size = x.size(0)
        if hs is None:
            h = Variable(t.zeros(config.num_layer, batch_size, config.hid_dim))
            c = Variable(t.zeros(config.num_layer, batch_size, config.hid_dim))
            hs = (h, c)
        if use_gpu:
            hs = (hs[0].cuda(), hs[1].cuda())
        out, hs_0 = self.lstm(x, hs)  # input：batch_size * train_len * input_dim；output：batch_size * train_len * hid_dim
        # out = out[:, -24:, :]
        out = out.contiguous()
        out = out.view(-1, config.hid_dim)  
        # normal net
        out = F.relu(self.bn(self.fc2(out)))
        # out = F.relu(self.fc4(out))
        out = self.fc3(out)

        return out, hs_0
```

---

## Requirements

The program is written in Python, and uses pytorch, scipy. A GPU is necessary, the ENN algorithm can only be running when CUDA is available.

- Install PyTorch(pytorch.org)
- Install CUDA
- `pip install -r requirements.txt`

## Training

- The training parameters like learning rate can be adjusted in configuration.py

```python
self.ERROR_PER = 0.02
self.path = 'E' + self.experiment_ID

self.GAMMA = 10
self.drop_last = False
# dataset setting
self.test_pro = 0.3
self.total_well_num = 14
self.train_len = int(24*4)  # length of the training data, 24*4 represents 4 days
self.predict_len = int(24 * 1)
# Network setting
self.T = 1
self.ne = 100
self.use_annual = 1 # the resolution of date information, whether to use the two-dimensional one-hot code (whether the date is in summer) as an input feature
self.use_quarterly = 0 # the resolution of date information, whether to use the four-dimensional one-hot code (corresponds to the four quarters) as an input feature
self.use_monthly = 0 # the resolution of date information, whether to use the 12-dimensional one-hot code (corresponds to the 12 monthes) as an input feature     
self.input_dim = 9  # annual:9; quarterly:12; monthly:20
self.hid_dim = 30  # number of hidden neurons
self.num_layer = 1  # number of LSTM layer in the network
self.drop_out = 0.3
self.output_dim = 1
# Trainting setting
self.batch_size = 512
self.num_workers = 0
self.learning_rate = 1e-3
self.weight_decay = 1e-4
self.display_step = 10
```

- To train a model, install the required module and run train_decay_loss.py, the result will be saved in the experiment folder.

```bash
python train_decay_loss.py
```







# The guide for EnLSTM:
Ensemble long short-term memory is a gradient-free neural network that combines ensemble neural network and long short-term memory.

This is part of implmentation of the paper [Ensemble long short-term memory (EnLSTM) network](https://arxiv.org/abs/2004.13562) by Yuantian Chen, Dongxiao Zhang, and Yuanqi Cheng.

It uses the ENN algorithm(see [Ensemble Neural Networks (ENN): A gradient-free stochastic method](https://www.sciencedirect.com/science/article/pii/S0893608018303319)) to train an artifical neural network instead of back propogation.

Notes: The EnLSTM code in this project is in the folder "enn". In this project, the word "enn" denotes the EnLSTM algorithm. Please refer to [EnLSTM codes](https://github.com/YuntianChen/EnLSTM) for more details. Besides, the codes of EnLSTM are also published on Zenodo Doi:10.5281/zenodo.413678.[Zenodo link](https://zenodo.org/account/settings/github/repository/YuntianChen/EnLSTM)

## Data Preperation and Model Definition

### Dataset Decription

The dataset used in this example contains data files formated as .csv files with header of feature names.

### Loading dataset

 ```python
 text = TextDataset()
```

### Neural Network Definition

```python
class netLSTM_withbn(nn.Module):

    def __init__(self):
        super(netLSTM_withbn, self).__init__()
        self.lstm = nn.LSTM(config.input_dim,
                            config.hid_dim,
                            config.num_layer,
                            batch_first=True,
                            dropout=config.drop_out)

        self.fc2 = nn.Linear(config.hid_dim,
                             int(config.hid_dim / 2))
        self.fc3 = nn.Linear(int(config.hid_dim / 2),
                             config.output_dim)
        self.bn = nn.BatchNorm1d(int(config.hid_dim / 2))

    def forward(self, x, hs=None, use_gpu=config.use_gpu):
        batch_size = x.size(0)
        if hs is None:
            h = Variable(t.zeros(config.num_layer,
                                 batch_size,
                                 config.hid_dim))
            c = Variable(t.zeros(config.num_layer,
                                 batch_size,
                                 config.hid_dim))
            hs = (h, c)
        if use_gpu:
            hs = (hs[0].cuda(), hs[1].cuda())
        out, hs_0 = self.lstm(x, hs)
        out = out.contiguous()
        out = out.view(-1, config.hid_dim)
        out = F.relu(self.bn(self.fc2(out)))
        out = self.fc3(out)
        return out, hs_0
```

---

## Requirements

The program is written in Python, and uses pytorch, scipy. A GPU is necessary, the ENN algorithm can only be running when CUDA is available.

- Install PyTorch(pytorch.org)
- Install CUDA
- `pip install -r requirements.txt`

## Training

- The training parameters like learning rate can be adjusted in configuration.py

```python
# training parameters
self.ne = 100
self.T = 1
self.batch_size = 32
self.num_workers = 1
self.epoch = 3
self.GAMMA = 10
```

- To train a model, install the required module and run train.py, the result will be saved in the experiment folder.

```bash
python train.py
```



# Model_files
For the model files, please refer to [model files](https://github.com/YuntianChen/EnLSTM/tree/main/model_files).

The folders in the “Model_files” are the supporting materials for the EnLSTM, including experiment results and model files for each experiment. There are five experiment folders and one Pytorch code in the “Model_files”. Each experiment file includes 14 trained neural network model files and three MSE loss files.

Regarding the trained neural network model files, which contain all the weights and bias in the EnLSTM. These files can be directly loaded in the ‘evaluate.py’ program and generate the corresponding models in Pytorch. People can use these models to predict and generate well logs. The file name contains 4 digits. The first digit indicates the group of experiments (in order to avoid the impact of randomness, all experiments are repeated five times), and the last two digits indicate the ID of the test well in the leave-one-out method. For example, model ‘1011’ means that the model is trained in the first group of experiment where the 11st well is taken as the test well.

Regarding the MSE loss, it contains the MSE loss of three epochs in a group of experiments. For example, exp1_epoch2 represents the MSE loss of the third epoch in the first experiment (epoch is calculated from 0). Each data file contains a matrix with size of 14*13. The 14 rows correspond to the 14 wells in the dataset, and the 13 columns are the 12 well logs and the average MSE. Specifically, the columns are Young’s modulus E_x and E_y, cohesion C, uniaxial compressive strength UCS, density ρ, tensile strength TS, brittleness index BI_x and BI_y, Poisson’s ratio ν_x and ν_y, neutron porosity NPR, total organic carbon TOC and average MSE from left to right.

Regarding the Pytorch code, it is named as “evaluate.py”. The model files can be directly loaded by this program and generate the corresponding models in Pytorch.

Notes: The trained neural network model files and the “evaluate.py” are all for the cascaded EnLSTM.
