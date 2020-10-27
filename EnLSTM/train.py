import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import pickle
import json
from enn import enn, enrml, lamuda
from net import netLSTM_withbn
from data import TextDataset
from configuration import config
from util import Record, save_var, get_file_list, list_to_csv, shrink, save_txt


# Set the random seed
LUCKY_NUM = 666666
torch.manual_seed(LUCKY_NUM)
torch.cuda.manual_seed(LUCKY_NUM)
np.random.seed(LUCKY_NUM)
# initialize matplotlib and CUDA
# plt.ion()
torch.cuda.set_device(config.deviceID)
# set the work path
PATH = config.path
if not os.path.isdir(PATH):
    os.makedirs(PATH)
# Parameters used in the net
ERROR_PER = config.ERROR_PER
NE = config.ne  # number of ensemble
GAMMA = config.GAMMA
T = config.T
# Load data and initialize enn net
text = TextDataset()
# Set the loss function
criterion = torch.nn.MSELoss()
INFO = {
    "train len": config.train_len,
    "shrink len": config.shrink_len,
    "window step": config.window_step,
    "Error per": config.ERROR_PER,
    "input dim": config.input_dim,
    "hid dim": config.hid_dim,
    "num layer": config.num_layer,
    "number of ensemble": config.ne,
    "T": config.T,
    "batch size": config.batch_size,
    "epoch": config.epoch,
    "GAMMA": config.GAMMA,
    "test ID": config.test_ID,
    "train ID": config.train_ID,
    "Weight": enn.ENN.W,
    "Lamuda": lamuda.Lamuda.L
}
with open('{}/Info.json'.format(PATH), 'w', encoding='utf-8') as f:
    json.dump(INFO, f, ensure_ascii=False)


# train the net, the result will be the net parameters and saved as pickle
def train(net_enn, input_, target, feature_name=''):
    dstb_y = lamuda.Lamuda(target, NE, ERROR_PER)
    train_losses = Record()
    losses = Record()
    lamuda_history = Record()
    std_history = Record()
    pred_history = Record()

    initial_parameters = net_enn.initial_parameters
    initial_pred = net_enn.output(input_)
    train_losses.update(criterion(initial_pred.mean(0), target).tolist())
    losses.update(criterion(initial_pred.mean(0), target).tolist())
    std_history.update(dstb_y.std(initial_pred))
    pred_history.update(initial_pred)
    lamuda_history.update(dstb_y.lamuda(initial_pred))

    for j in range(T):
        torch.cuda.empty_cache()
        params = net_enn.get_parameter()
        dstb_y.update()
        time_ = time.strftime('%Y%m%d_%H_%M_%S')
        delta = enrml.EnRML(pred_history.get_latest(mean=False), params, initial_parameters,
                            lamuda_history.get_latest(mean=False), dstb_y.dstb, ERROR_PER)
        params_raw = net_enn.update_parameter(delta)
        torch.cuda.empty_cache()
        pred = net_enn.output(input_)
        loss_new = criterion(pred.mean(0), target).tolist()
        bigger = train_losses.check(loss_new)
        record_while = 0
        while bigger:
            record_while += 1
            lamuda_history.update(lamuda_history.get_latest(mean=False) * GAMMA)
            if lamuda_history.get_latest(mean=False) > GAMMA ** 10:
                lamuda_history.update(lamuda_history.data[0])
                print('abandon current iteration')
                net_enn.set_parameter(params)
                loss_new = train_losses.get_latest()
                dstb_y.update()
                params_raw = params
                break
            dstb_y.update()
            net_enn.set_parameter(params)
            delta = enrml.EnRML(pred_history.get_latest(mean=False), params, initial_parameters,
                                lamuda_history.get_latest(mean=False), dstb_y.dstb, ERROR_PER)
            params_raw = net_enn.update_parameter(delta)
            torch.cuda.empty_cache()
            pred = net_enn.output(input_)
            loss_new = criterion(pred.mean(0), target).tolist()
            print('update losses, new loss:{}'.format(loss_new))
            bigger = train_losses.check(loss_new)
        train_losses.update(loss_new)
        save_var(params_raw, '{}/{}_{}_params'.format(PATH, time_, feature_name))
        print("iteration:{} \t current train losses:{}".format(j, train_losses.get_latest(mean=True)))
        save_txt('{}/loss_{}.txt'.format(PATH, feature_name), time.strftime('%Y%m%d_%H_%M_%S')+','+str(train_losses.get_latest(mean=True))+',\n')
        pred_history.update(pred)
        std_history.update(dstb_y.std(pred))
        if std_history.bigger():
            lamuda_history.update(lamuda_history.get_latest(mean=False))
        else:
            lamuda_tmp = lamuda_history.get_latest(mean=False) / GAMMA
            if lamuda_tmp < 0.005:
                lamuda_tmp = 0.005
            lamuda_history.update(lamuda_tmp)
    return net_enn, train_losses.get_latest(mean=True), pred_history.get_latest(mean=False)


# predict in one calculation
def predict_full(data, params=None, model_predict=None):
    input_ = torch.tensor(data)
    input_ = Variable(input_.view(1, len(data), config.input_dim).float()).cuda()
    if params is not None:
        model_predict.set_parameter(params)
    pred = model_predict.output(input_)
    return pred


# draw the result based on the latest parameters
def draw_result(enn_net):
    # Get the latest parameters, and initialize the enn net
    param_list = get_file_list('params', config.path)
    params = pickle.load(open(param_list[-1], 'rb'))
    print("use parameter file: {}".format(param_list[-1]))
    enn_net.set_parameter(params)
    # Draw the result of well in test well lists
    for well in config.test_ID:
        input_, target = text.test_dataset(well)
        pred_enn = predict_full(input_, params=params, model_predict=enn_net).cpu()
        # output the loss
        loss = criterion(pred_enn.mean(0), torch.tensor(target).float())
        print("well{}\t test loss: {}".format(well, loss))
        # get the real predicted and target data
        pred = text.inverse_normalize(pred_enn.mean(0))
        target = text.inverse_normalize(target)
        std = 3 * np.array(text.inverse_normalize(pred_enn).std(0))
        # save the test loss
        save_txt('{}/test_loss.txt'.format(PATH), '{}, {}\n'.format(loss, std.mean()))
        print('std:', std.mean())
        x = np.arange(len(target))
        plt.figure(figsize=(60, 5))
        plt.plot(target, label='target', color='black', alpha=0.4)
        plt.errorbar(x, pred[:, 0], yerr=std[:, 0], color='red', alpha=0.7)
        plt.title(config.info)
        plt.legend()
        ylabel = config.columns[config.input_dim+1]
        plt.ylabel(ylabel)
        plt.savefig('{}/result.png'.format(PATH))
        plt.show()


def test(enn_net, feature_name='', draw_result=False):
    # Get the latest parameters, and initialize the enn net
    param_list = get_file_list('{}_params'.format(feature_name), config.path)
    params = pickle.load(open(param_list[-1], 'rb'))
    print("use parameter file: {}".format(param_list[-1]))
    enn_net.set_parameter(params)
    # Draw the result of well in test well lists
    for well in config.test_ID:
        input_, target = text.test_dataset(well)
        pred_enn = predict_full(input_, params=params, model_predict=enn_net).cpu()
        # output the loss
        loss = criterion(pred_enn.mean(0), torch.tensor(target).float())
        print("well{}\t{}\ttest loss: {}".format(well, feature_name, loss))
        # replace the test dataset and reset train dataset
        text.df_list[well-1][[feature_name]] = np.array(text.inverse_normalize(pred_enn.mean(0)))
        # get the std
        std = 3 * np.array(text.inverse_normalize(pred_enn).std(0))
        # save the test loss
        save_txt('{}/test_loss_{}.txt'.format(PATH, feature_name), '{}, {}, {}\n'.format(feature_name, loss, std.mean()))
        print('std:', std.mean())
        if draw_result:
            # get the real predicted and target data
            pred = np.array(text.inverse_normalize(pred_enn.mean(0)))
            target = text.inverse_normalize(target)
            x = np.arange(len(target))
            plt.figure(figsize=(60, 5))
            plt.plot(target, label='target', color='black', alpha=0.4)
            plt.errorbar(x, pred[:, 0], yerr=std[:, 0], color='red', alpha=0.7)
            plt.title(config.info)
            plt.legend()
            y_label = feature_name
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.savefig('{}/result_{}.png'.format(PATH, feature_name))
            # plt.show()


def run():
    for epoch in range(config.epoch):
        print(epoch)
        while config.input_dim+1 <= len(config.columns):
            current_feature_name = config.columns[config.input_dim]
            textLoader = DataLoader(text, batch_size=config.batch_size, shuffle=True,
                                    num_workers=config.num_workers, drop_last=config.drop_last)
            model = netLSTM_withbn()
            with torch.no_grad():
                model = model.cuda()
            net_enn_train = enn.ENN(model, NE)
            # If pre_existent epoch found, set net_enn_train parameters with pre_existent epoch record.
            # Only processed if current epoch count is 0
            epoch_list = [i for i in os.listdir(PATH) if i.startswith(
                "parameters_{}_epoch_".format(current_feature_name))]
            if len(epoch_list) > 0 and epoch == 0:
                print("Pre_existent epoch found: {}".format(sorted(epoch_list)[-1]))
                epoch_pre_existent = pickle.load(open(os.path.join(PATH, sorted(epoch_list)[-1]), 'rb'))
                net_enn_train.set_parameter(epoch_pre_existent)
            if epoch > 0:
                parameter_path = os.path.join(PATH, "parameters_{}_epoch_{}".format(current_feature_name, epoch-1))
                print("Setting checkpoint {}".format(parameter_path))
                parameter_checkpoint = pickle.load(open(parameter_path, 'rb'))
                net_enn_train.set_parameter(parameter_checkpoint)
            for i, data in enumerate(textLoader):
                print('#'*30)
                print("{}: batch{}".format(time.strftime('%Y%m%d_%H_%M_%S'), i))
                # preparing the train data
                input_, target = data
                input_ = torch.from_numpy(np.stack(list(shrink(input_, config.shrink_len)), axis=1))
                target = torch.from_numpy(np.stack(list(shrink(target, config.shrink_len)), axis=1))
                with torch.no_grad():
                    input_, target = map(Variable, (input_.float(), target.float()))
                    target = target.reshape(-1, config.output_dim)
                    input_ = input_.cuda()
                    target = target.cuda()
                # train the model
                net_enn_train, loss, pred_data = train(net_enn_train, input_, target, feature_name=current_feature_name)
            with torch.no_grad():
                params = net_enn_train.get_parameter()
                filename = PATH+"/parameters_{}_epoch_{}".format(current_feature_name, epoch)
                save_var(params, filename)
                del params
            test(net_enn_train, feature_name=current_feature_name, draw_result=(epoch == config.epoch-1))
            config.input_dim += 1
            text.reset_train_dataset()
        config.input_dim -= 3
        text.reset_train_dataset()
        text.reset_test_dataset()


if __name__ == '__main__':
    run()

