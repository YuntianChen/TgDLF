import torch as t
import torch.nn as nn
from torch.autograd import Variable
from configuration import config
import torch.nn.functional as F  

class netLSTM(nn.Module):
    
    def __init__(self):
        super(netLSTM, self).__init__()
        self.lstm = nn.LSTM(config.input_dim,
                            config.hid_dim, 
                            config.num_layer, 
                            batch_first=True,
                            dropout=config.drop_out)
        self.fc2 = nn.Linear(config.hid_dim,
                             int(config.hid_dim/2))
        self.fc3 = nn.Linear(int(config.hid_dim/2),
                             config.output_dim)

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
        self.lstm.flatten_parameters()
        out, hs_0 = self.lstm(x, hs)
        out = out[:, -10:, :]
        self.lstm.flatten_parameters()
        out = out.contiguous()
        out = out.view(-1, config.hid_dim) 
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out, hs_0


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

