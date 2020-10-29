from enn import *
import numpy as np
from grid_LSTM import netLSTM, netLSTM_full
from grid_data_v2 import TextDataset
import grid_data_v2 as grid_data
from grid_configuration import config
from util import Record, save_var, get_file_list, Regeneralize, list_to_csv
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import time
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
plt.ion()

seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
    
torch.cuda.set_device(config.deviceID)
print(config.test_ID)
PATH = config.path
if not os.path.exists(PATH):
    os.mkdir(PATH)
'''Parmaters used in the net.'''
ERROR_PER = config.ERROR_PER
NE = config.ne  # number of ensemble
GAMMA = config.GAMMA
T = config.T

''' load data and initialize enn net'''
text = TextDataset(config)
textLoader = DataLoader(text, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers, drop_last=config.drop_last)
criterion = torch.nn.MSELoss()


def train(net_enn, input_, target):
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
            if lamuda_history.get_latest(mean=False) > GAMMA ** 5:
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
        save_var(params_raw, '{}/{}_params'.format(PATH, time_))
        print("iteration:{} \t current train losses:{}".format(j, train_losses.get_latest(mean=True)))
        with open('{}/loss.txt'.format(PATH), 'a') as f:
            f.write(time.strftime('%Y%m%d_%H_%M_%S')+','+str(train_losses.get_latest(mean=True))+',\n')
        f.close()
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


def predict(data, params=None, model_predict=None):
    result = []
    input_ = torch.tensor(data)
    input_ = Variable(input_.view(1, len(data), config.input_dim).float()).cuda()
    if params is not None:
        model_predict.set_parameter(params)
    i = 0
    while i <= len(data) - config.train_len:
        pred = model_predict.output(input_[:, i:i+config.train_len, :])
        result.append(pred[:, -24:, :])
        print('predicting: {} to {}'.format(i, i + config.train_len))
        i += 24
    #save_var(result, 'result')
    return torch.cat(result, dim=1)


def predict_full(data, params=None, model_predict=None):
    input_ = torch.tensor(data)
    input_ = Variable(input_.view(1, len(data), config.input_dim).float()).cuda()
    if params is not None:
        model_predict.set_parameter(params)
    pred = model_predict.output(input_)
    return pred


def draw_result(enn_net):
    param_list = get_file_list('params', config.path)
    params = pickle.load(open(param_list[-1], 'rb'))
    print("use parameter file: {}".format(param_list[-1]))
    enn_net.set_parameter(params)
    for i, k in enumerate(config.test_ID):
        input_ = text.test_data_input_list[i]
        target = text.test_data_output_list[i]
        raw_data = pd.read_csv("data/{}.csv".format(config.data_list[k-1]))
        real_std = raw_data.LOAD.std()
        real_mean = raw_data.LOAD.mean()
        raw = np.array(raw_data.LOAD)[config.predict_len:-config.predict_len]
        pred = predict_full(input_, params=params, model_predict=enn_net)

        # 平移,baseline
        #pred = np.zeros((config.ne, np.shape(raw)[0], 1))#趋势平移
        #pred = np.ones((config.ne, np.shape(raw)[0], 1)) #纯粹平移
        
        # save the result right from the enn net
        np.savetxt('result/e{}-p{}-pred_w{}.csv'.format(config.experiment_ID, PATH, k),
                   np.array(pred)[:, :, 0].T, delimiter=',')  
        # 加上平均趋势，获得真正的ratio      
        if grid_data.use_mean_ratio == True:
            if grid_data.use_different_mean_ratio == True:
                mean_ratio = text.mean_ratio_all_pred[k-1,:]
            elif grid_data.use_CV_ratio == True:
                mean_ratio = text.mean_ratio_group_pred[config.test_set_ID-1,:]
            else:
                mean_ratio = text.mean_ratio_all_ave_pred
            print('mean_ratio:', mean_ratio)
            pred = np.array(pred) + mean_ratio.reshape(-1,1) # 补充上平均趋势，ave表示这是所有区平均后的趋势
            #test1 = mean_ratio.reshape(-1,1)
            #pred=np.array([list(test1) for i in range(100)])
            np.savetxt('result/e{}-p{}-pred_mean_ratio_w{}.csv'.format(config.experiment_ID, PATH, k),
                    np.array(pred)[:, :, 0].T, delimiter=',')  
            print('test_pred:', np.shape(pred))
            target = target + mean_ratio.reshape(-1,1)      
            loss = criterion(torch.tensor(pred.mean(0)[:, 0]).float().cpu(), torch.tensor(target[:, 0]).float())
        else:
            loss = criterion(torch.tensor(pred.mean(0)[:, 0]).float().cpu(), torch.tensor(target[:, 0]).float())
        print("ID{}\t test loss: {}".format(k, loss))
        mean = grid_data.load_mean[k - 1][0]
        std = grid_data.load_std[k - 1][0]
        pred_ratio = Regeneralize(np.array(pred[:, :, 0]), mean, std)
        pred_real = pred_ratio * raw
        #pred_real = pred_ratio
        target_ratio = Regeneralize(target, mean, std).reshape(-1,1)
        target_real = target_ratio * raw.reshape(-1,1)
        #target_real = target_ratio
        loss_ratio = criterion(torch.tensor(pred_ratio.mean(0)).float(), torch.tensor(target_ratio[:, 0]).float()) #new
        print("ID{}\t ratio loss: {}".format(k, loss_ratio))
        #target_real = np.array(raw_data.LOAD)[config.predict_len*2:]
        # make a normalization of real load value:
        loss_relative = np.mean(np.abs(pred_real.mean(0) - target_real.reshape(-1))/target_real.reshape(-1))
        std = 1 * pred_real.std(0)
        pred_normalized = (pred_real.mean(0) - real_mean) / real_std
        target_normalized = (target_real.reshape(-1) - real_mean) / real_std
        print('pred_normalized shape:', np.shape(pred_normalized))
        print('target_normalized shape:', np.shape(target_normalized))
        loss_real = criterion(Variable(torch.tensor(pred_normalized).float()),
                              Variable(torch.tensor(target_normalized).float()))
        print("ID{}\t relative loss: {}".format(k, loss_relative))
        print("ID{}\t real loss: {}".format(k, loss_real))
        with open('{}/test_loss_{}.csv'.format(PATH, config.experiment_ID), 'a') as f:
            f.write('{},{},{},{},{},{}\n'.format(k, loss, loss_ratio, loss_real, loss_relative, std.mean()))
        f.close()
        print('std:', std.mean())
        x = np.arange(len(target))
        np.savetxt('result/e{}-p{}-pred_w{}_real.csv'.format(config.experiment_ID, PATH, k),
                   np.array(pred_real).T, delimiter=',')
        print('flag1')
        np.savetxt('result/e{}-p{}-target_w{}_real.csv'.format(config.experiment_ID, PATH, k), 
                   target, delimiter=',')
        print('Plotting')
        plt.figure(figsize=(100, 5))
        plt.plot(target_real, label='target', color='black', alpha=0.4)
        plt.errorbar(x, pred_real.mean(0), yerr=std, color='red', alpha=0.7)
        plt.title(str(k) + '-' + config.info)
        plt.legend()
        plt.savefig('{}/ID{}.png'.format(PATH, k))
        plt.show()
        print('flag2')
        
        
        
def evaluate(enn_net, epoch):
    
    #params = os.path.join(config.path, 'parameters_epoch%d' % epoch)
    param_list = get_file_list('params', config.path)
    #param_index = int(len(param_list)*epoch/config.epoch)-1
    param_index = int(len(param_list)*(epoch+1)/config.epoch)-1
    print('total number of saved parameters: %d, using no %d' % (len(param_list), param_index))
    params = param_list[param_index]
    print("use parameter file: {}".format(params))
    params = pickle.load(open(params, 'rb'))
    enn_net.set_parameter(params)
    for i, k in enumerate(config.test_ID):
        input_ = text.test_data_input_list[i] # ->array(data_len * in_dim)
        target = text.test_data_output_list[i]# ->array(data_len * 1)
        raw_data = pd.read_csv("data/{}.csv".format(config.data_list[k-1]))
        real_std = raw_data.LOAD.std()
        real_mean = raw_data.LOAD.mean()
        raw = np.array(raw_data.LOAD)[config.predict_len:-config.predict_len]
        pred = predict_full(input_, params=params, model_predict=enn_net)# ->tensor(ensemble_size*data_len*1)

        # 平移,baseline
        #pred = np.zeros((config.ne, np.shape(raw)[0], 1))#趋势平移
        #pred = np.ones((config.ne, np.shape(raw)[0], 1)) #纯粹平移
        
        # save the result right from the enn net
        np.savetxt('result/e{}-epoch{}-pred_w{}.csv'.format(config.experiment_ID, epoch, k),
                   np.array(pred)[:, :, 0].T, delimiter=',')  
        # 加上平均趋势，获得真正的ratio      
        if grid_data.use_mean_ratio == True:
            if grid_data.use_different_mean_ratio == True:
                mean_ratio = text.mean_ratio_all_pred[k-1,:]
            elif grid_data.use_CV_ratio == True:
                mean_ratio = text.mean_ratio_group_pred[config.test_set_ID-1,:]                
            else:
                mean_ratio = text.mean_ratio_all_ave_pred
            print('mean_ratio:', mean_ratio)
            pred = np.array(pred) + mean_ratio.reshape(-1,1) # 补充上平均趋势，ave表示这是所有区平均后的趋势
            #test1 = mean_ratio.reshape(-1,1)
            #pred=np.array([list(test1) for i in range(100)])
            np.savetxt('result/e{}-epoch{}-pred_mean_ratio_w{}.csv'.format(config.experiment_ID, epoch, k),
                    np.array(pred)[:, :, 0].T, delimiter=',')  
            print('test_pred:', np.shape(pred))
            target = target + mean_ratio.reshape(-1,1)      
            loss = criterion(torch.tensor(pred.mean(0)[:, 0]).float().cpu(), torch.tensor(target[:, 0]).float())
        else:
            loss = criterion(torch.tensor(pred.mean(0)[:, 0]).float().cpu(), torch.tensor(target[:, 0]).float())
        print("ID{}\t test loss: {}".format(k, loss))
        mean = grid_data.load_mean[k - 1][0]
        std = grid_data.load_std[k - 1][0]
        pred_ratio = Regeneralize(np.array(pred[:, :, 0]), mean, std)
        pred_real = pred_ratio * raw
        #pred_real = pred_ratio
        target_ratio = Regeneralize(target, mean, std).reshape(-1,1)
        target_real = target_ratio * raw.reshape(-1,1)
        #target_real = target_ratio
        loss_ratio = criterion(torch.tensor(pred_ratio.mean(0)).float(), torch.tensor(target_ratio[:, 0]).float()) #new
        print("ID{}\t ratio loss: {}".format(k, loss_ratio))
        #target_real = np.array(raw_data.LOAD)[config.predict_len*2:]
        # make a normalization of real load value:
        loss_relative = np.mean(np.abs(pred_real.mean(0) - target_real.reshape(-1))/target_real.reshape(-1))
        std = 1 * pred_real.std(0)
        pred_normalized = (pred_real.mean(0) - real_mean) / real_std
        target_normalized = (target_real.reshape(-1) - real_mean) / real_std
        print('pred_normalized shape:', np.shape(pred_normalized))
        print('target_normalized shape:', np.shape(target_normalized))
        loss_real = criterion(Variable(torch.tensor(pred_normalized).float()),
                              Variable(torch.tensor(target_normalized).float()))
        print("ID{}\t relative loss: {}".format(k, loss_relative))
        print("ID{}\t real loss: {}".format(k, loss_real))
        f =  open(r'{}/epoch{}_test_loss_{}.csv'.format(PATH, epoch, config.experiment_ID), 'a')
        f.write('{},{},{},{},{},{}\n'.format(k, loss, loss_ratio, loss_real, loss_relative, std.mean()))
        f.close()
        print('std:', std.mean())
        x = np.arange(len(target))
        np.savetxt('result/e{}-epoch{}-pred_w{}_real.csv'.format(config.experiment_ID, epoch, k),
                   np.array(pred_real).T, delimiter=',')
        print('flag1')
        np.savetxt('result/e{}-epoch{}-target_w{}_real.csv'.format(config.experiment_ID, epoch, k), 
                   target, delimiter=',')
        print('Plotting')
        plt.figure(figsize=(100, 5))
        plt.plot(target_real, label='target', color='black', alpha=0.4)
        plt.errorbar(x, pred_real.mean(0), yerr=std, color='red', alpha=0.7)
        plt.title(str(k) + '-' + config.info)
        plt.legend()
        plt.savefig('{}/ID{}_epoch{}.png'.format(PATH, k, epoch))
        plt.show()
        print('flag2')
    

def save_result(enn_net):
    test_loss = []
    for i, k in enumerate(config.test_ID):
        input_ = text.test_data_input_list[i]
        target = text.test_data_output_list[i]
        pred = predict_full(input_, params=None, model=enn_net)
        loss = criterion(pred.cpu(), torch.tensor(target).float())
        test_loss.append(loss)
    with open(PATH + '/test_loss.txt', 'a') as f:
        f.write(time.strftime('%Y%m%d_%H_%M_%S') + ',' + str(test_loss) + ',\n')
    f.close()


def run():
    with open('{}/time.txt'.format(PATH), 'a') as f:
        f.write('{},\n'.format(time.strftime('%Y%m%d_%H_%M_%S')))
    f.close()
    model = netLSTM()
    with torch.no_grad():
        model = model.cuda()
    net_enn_train = enn.ENN(model, NE)
    for epoch in range(config.epoch):
        for i, data in enumerate(textLoader):
            print('#'*30)
            print("{}: batch{}".format(time.strftime('%Y%m%d_%H_%M_%S'), i))
            input_, target = data
            #input_ = torch.from_numpy(np.stack(list(shrink(input_, 5)), axis=1))
            #target = torch.from_numpy(np.stack(list(shrink(target, 5)), axis=1))
            with torch.no_grad():
                input_, target = map(Variable, (input_.float(), target.float()))
                target = target[:, -config.predict_len:, :]
                print(target.shape)
                target = target.reshape(-1, config.output_dim)
                input_ = input_.cuda()
                target = target.cuda()
            net_enn_train, loss, pred_data = train(net_enn_train, input_, target)
            
            # save pred and target while training
            save_dir = os.path.join(PATH, 'predict_history')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_data = {}
            save_data['pred'] = np.array(pred_data.mean(0)[:, 0])
            save_data['target'] = np.array(np.array(target[:, 0]))
            save_data = pd.DataFrame.from_dict(save_data)
            save_filename = '{}_{}.csv'.format(epoch, i)
            save_data.to_csv(os.path.join(save_dir, save_filename))
            """
            with open('predict_history'+'/pred.txt', 'a') as f:
                f.write(list_to_csv(np.array(pred_data.mean(0)[:, 0])) + '\n')
            f.close()
            with open('predict_history'+'/target.txt', 'a') as f:
                f.write(list_to_csv(np.array(target[:, 0])) + '\n')
            f.close()
            """
            with open(PATH+'/time.txt', 'a') as f:
                f.write(time.strftime('%Y%m%d_%H_%M_%S') + ',' + str(loss) + ',\n')
            f.close()
        with torch.no_grad():
            params = net_enn_train.get_parameter()
            filename = PATH+"/parameters_epoch{}".format(epoch)
            save_var(params, filename)
            del params


if __name__ == '__main__':
    #run()#
    model = netLSTM_full()
    with torch.no_grad():
        model = model.cuda()
    net_enn = enn.ENN(model, NE)
    #draw_result(net_enn)#
    evaluate(net_enn, 4)# count from 0
    print(config.test_ID)

