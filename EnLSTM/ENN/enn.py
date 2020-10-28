import torch
import numpy as np


class ENN:
    W = 1
    
    def __init__(self, torch_model, numberofensemble):
        self.Ne = numberofensemble
        ####################################################
        # self.param_list: parameter name list of model e.g. [layer1.weight, layer1.bias, layer2.weight, layer.bias]
        # self.param_index: store the location of parameters in a 1-d list.
        # e.g. [[1,3],[3,5],[5,9], the [1,3] shows that the first parameters start from 1 and end at 3 in the 1-d list
        # self.param_size: the shape of parameters,e.g. layer1.weight:3x4,which is [3,4]
        # self.Nw: sum of parameters
        self.param_list, self.param_index, self.param_size, self.Nw = self.count_parameters(torch_model)
        self.parameters = torch.zeros(self.Nw, self.Ne, requires_grad=False, dtype=torch.float32)
        self.initial_parameters = torch.randn(self.Nw, self.Ne, requires_grad=False, dtype=torch.float32)  # initialize the parameters
        self.update_parameter(self.initial_parameters)
        self.model = torch_model

    def output(self, in_feature):
        output = []
        for param in self.parameters.t():
            with torch.no_grad():
                param = param.cuda()
            for i, name in enumerate(self.param_list):  # set the model parameters layer by layer
                head = self.param_index[i][0]
                tail = self.param_index[i][1]
                size = self.param_size[i]
                parameters = param[head:tail].reshape(size)
                self.model.state_dict()[name] -= self.model.state_dict()[name] - parameters
            with torch.no_grad():
                out, hs = self.model(in_feature)
            # out = out[-10:, :] # get just 1 meter data
            output.append(out)
            # del model, param
            torch.cuda.empty_cache()
            # print(out)
        result = torch.stack(output)  # type: torch.tensor size:Ne*batch*feature
        # print(result.mean(0).mean(0))
        return result # the output of enn is a 3-d tensor whith the first dimension is Ne

    @staticmethod
    def count_parameters(m):  # return the structure of parameters in model
        param_list = list(m.state_dict())[:-3]
        # param_list = list(m.state_dict())
        total_param = 0
        param_index = []
        size = []
        j = 0
        for name, param in m.named_parameters():
            size.append(list(param.size()))
            num_param = np.prod(param.size())
            i = j
            j += num_param
            total_param += num_param
            param_index.append([i, j])
        return param_list, param_index, size, int(total_param)

    def get_parameter(self):
        return self.parameters

    def update_parameter(self, delta):  # update the parameters
        parameters_raw = self.parameters + delta
        self.parameters = self.add_var(parameters_raw)
        return parameters_raw

    def reset_parameter(self):
        self.parameters = torch.randn(self.Nw, self.Ne)

    def set_parameter(self, value):
        self.parameters = value

    def add_var(self, value):
        # delta_randn = torch.tensor(SIGMA * np.random.randn(self.Nw, self.Ne)).float()
        value_mean = value.mean(1).reshape(-1, 1)
        value_std = value.std(1).reshape(-1, 1)
        value = 0.99 * value + 0.01 * value_mean + value_std * self.W * torch.randn(self.Nw, self.Ne)
        return value
