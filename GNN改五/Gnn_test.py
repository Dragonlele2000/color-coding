# GNN主体
import math
import threading
from threading import Timer

from pynput.keyboard import Controller, Key, Listener

from Ideal import root
from config_test import *
import random
import torch
from torch.nn import Linear, BatchNorm1d, BatchNorm2d
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
from GetDataset import get_dataset
import torch.nn as nn
import numpy as np


class GConv(torch.nn.Module):
    def __init__(self):
        super(GConv, self).__init__()
        torch.manual_seed(randomSeed)
        self.tuning_step = tuning_step
        if use_linear:
            self.linear = nn.Linear(num_node_features, 1, bias=False)
        else:
            self.e = torch.zeros((1, num_node_features))
            if GPU:
                self.e = self.e.cuda()
            for i in range(k):
                self.e[0][(i + 1) * (1 << k) - 1] = 1
        if use_max:
            self.conv1 = GraphConv(num_node_features, num_node_features, aggr='max', bias=False)
        else:
            self.conv1 = GraphConv(num_node_features, num_node_features, aggr='add', bias=False)
        #改动
        # self.conv1.lin_rel.weight.data=2*self.conv1.lin_rel.weight.data
        # torch.nn.init.normal_(self.conv1.lin_rel.weight, 0.2, 0.2)
        # torch.nn.init.normal_(self.conv1.lin_root.weight, -1, 1)
        # torch.nn.init.kaiming_uniform_(self.conv1.lin_rel.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_uniform_(self.conv1.lin_root.weight, nonlinearity='relu')
        if fix_root:
            # torch.nn.init.normal_(self.conv1.lin_rel.weight, 1, .05)
            self.conv1.lin_root.weight.data = torch.FloatTensor(root(k))
            # self.conv1.lin_root.requires_grad_(False)
            # torch.nn.init.normal_(self.conv1.lin_root.weight, -30, 1)

        # torch.nn.init.normal_(self.conv1.lin_root.weight,1,0.1)
        # torch.nn.init.normal_(self.conv1.lin_rel.weight,0.2,0.05)

    def forward(self, x, edge_index, batch):
        for i in range(k - 1):
            x = self.conv1(x, edge_index)
            x = x.relu()
        # x = self.lin(x)
        if use_linear:
            x = self.linear(x)
        else:
            x = (torch.matmul(self.e, x.T)).T
        return x


count = 0


def train(model, train_loader, optimizer, criterion):
    model.train()
    global count
    count += 1
    avg_loss = 0
    if not fix_root:
        w_root = model.conv1.lin_root.weight.data
        model.conv1.lin_root.weight.data = torch.where((upper_bound < w_root),
                                                       w_root * step_div, w_root)
        w_root = model.conv1.lin_root.weight.data
        model.conv1.lin_root.weight.data = torch.where((lower_bound < w_root) & (w_root < threshold),
                                                       w_root * step_mul - step, w_root)
    # if tuning_rel:
    #     w_rel = model.conv1.lin_rel.weight.data
    #     model.conv1.lin_rel.weight.data = torch.where((w_rel <= tuning_threshold),
    #                                                   w_rel * mul_tuning_step + tuning_step, w_rel)
    #     w_rel = model.conv1.lin_rel.weight.data
    #     model.conv1.lin_rel.weight.data = torch.where((w_rel > tuning_threshold2),
    #                                                   (w_rel - 1) * mul_tuning_step2 + 1 - tuning_step2, w_rel)

    for data in train_loader:
        if tuning_rel:
            w_rel = model.conv1.lin_rel.weight.data
            model.conv1.lin_rel.weight.data = torch.where((w_rel <= tuning_threshold),
                                                          w_rel * mul_tuning_step + tuning_step, w_rel)
            w_rel = model.conv1.lin_rel.weight.data
            model.conv1.lin_rel.weight.data = torch.where((w_rel > tuning_threshold2),
                                                          (w_rel - 1) * mul_tuning_step2 + 1 - tuning_step2, w_rel)


        # if train_count % 10000 == 0:
        #    print('training:' + str(train_count))
        if GPU:
            data = data.cuda()
        out = model(data.x, data.edge_index, data.batch)
        out = out.reshape(-1)

        loss = criterion(out, data.y)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = avg_loss / len(train_loader)
    if count % print_interval == 0:
        print('\033[0;31m' + f'loss: {avg_loss:.10f}' + '\033[0m'
              + f', avg_root={model.conv1.lin_root.weight.data.mean().item():.5f}'
              + f', num_threshold={(model.conv1.lin_root.weight.data <= threshold).sum().item():.0f}'
              + f', num_lower_bound={(model.conv1.lin_root.weight.data <= lower_bound).sum().item():.0f}'
              + f', rel_max_element={model.conv1.lin_rel.weight.data.max().item():.6f}'
              + f', num_rel_element > 0.5:{(model.conv1.lin_rel.weight.data > 0.5).sum().item():.0f}'
              + f', rel_min_element={model.conv1.lin_rel.weight.data.min().item():.6f}'
              + f', num_rel_element < 0.01:{(model.conv1.lin_rel.weight.data < 0.01).sum().item():.0f}'
              + f'')
    return avg_loss


def test(model, test_loader):
    model.eval()

    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    # data_no=0
    for data in test_loader:
        if GPU:
            data = data.cuda()
        out = model(data.x, data.edge_index, data.batch)
        out = out.reshape(-1)

        total += out.numel()
        result = abs(out - data.y)
        correct1 += (result < correct_standard_1).sum().item()
        correct2 += (result < correct_standard_2).sum().item()
        correct3 += (result < correct_standard_3).sum().item()
    return correct1 / total, correct2 / total, correct3 / total


reset_count_down = False

class my_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        temp=abs(x-y)
        temp=torch.where(temp>-1e-4,temp,pow(temp, 2))
        return torch.mean(temp)
        #return torch.mean(torch.pow(x - y, 2))

def on_press(key):
    global reset_count_down
    if str(key) == r"'\x03'":  # ctrl C
        reset_count_down = True


def thread_run():
    with Listener(on_press=on_press) as listener:
        listener.join()


def main():
    # keyboard = Controller()
    thread = threading.Thread(target=thread_run)
    thread.start()
    if GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print('Getting dataset')
    train_dataset, test_dataset = get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    print('Loading model')
    model = GConv()
    params_dict = [{'params': model.conv1.lin_rel.parameters(), 'lr': learning_rate_rel},
                   {'params': model.conv1.lin_root.parameters(), 'lr': learning_rate_rel * rate_ratio}]

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(params_dict)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    #criterion = my_loss()
    if GPU:
        model = model.cuda()
        criterion = criterion.cuda()

    ratio = rate_ratio

    rate_rel = learning_rate_rel
    rate_root = learning_rate_rel * ratio
    num_drop = 0
    count_down = 0
    for epoch in range(1, num_epoch + 1):
        if count_down == 0:
            hasException = True
            while hasException:
                try:
                    pack = input(f'输入训练次数 学习率(当前{rate_rel:.6f}) ratio(当前{ratio:.0f})：').split(' ')
                    if len(pack) == 3:
                        ratio = eval(pack[2])
                        pack = pack[:-1]
                    if len(pack) == 2:
                        count_down = eval(pack[0])
                        new_rate = eval(pack[1])
                        hasException = False
                        rate_rel = new_rate
                        rate_root = rate_rel * ratio
                        optimizer.param_groups[0]['lr'] = rate_rel
                        optimizer.param_groups[1]['lr'] = rate_root
                    elif len(pack) == 1:
                        if pack[0] == 'reset':
                            model.conv1.lin_rel.reset_parameters()
                            #torch.nn.init.normal_(model.conv1.lin_rel.weight.data,0.5,0.05)
                            if not fix_root:
                                model.conv1.lin_root.reset_parameters()
                            continue
                        elif pack[0].split(',')[0] == 'tuning':
                            model.tuning_step = eval(pack[0].split(',')[1])
                            continue
                        count_down = eval(pack[0])
                        hasException = False
                except Exception as e:
                    print('输入格式错误')
                    print(f'{e}')
                    hasException = True
        else:
            global reset_count_down
            if reset_count_down:
                count_down = 0
                reset_count_down = False
                continue
        count_down -= 1

        # 手动调整root
        if not fix_root:
            if enable_drop:
                w_rel = model.conv1.lin_root.weight.data
                drop = w_rel <= threshold
                new_num_drop = drop.sum()
                if new_num_drop > num_drop:
                    num_drop = new_num_drop
                    model.conv1.lin_root.reset_parameters()
                    # model.conv1.lin_rel.reset_parameters()
                    w_rel = model.conv1.lin_root.weight.data
                    model.conv1.lin_root.weight.data = torch.where(drop, lower_bound * 10, w_rel)
                    # torch.nn.init(model.conv1.lin_root.weight.data)
                    # model.conv1.lin_root.weight.data = torch.where((lower_bound < w_rel) & (w_rel < threshold),
                    # w_rel * step_mul - step, w_rel)
            else:
                # w_rel = model.conv1.lin_root.weight.data
                # model.conv1.lin_root.weight.data = torch.where((lower_bound < w_rel) & (w_rel < threshold),
                #     w_rel * step_mul - step, w_rel)
                # optimizer.zero_grad()
                changed = 1
        if epoch % print_interval == 0:
            print('-' * 50)
            print(f'Epoch: {epoch:03d},countdown: {count_down:.0f},'
                  f'(Ctrl+C 暂停,reset 重置参数, tuning+\',\'+step 设置回调)')
            print(f'learning_rate:{rate_rel:.6f}, {rate_root:.6f}, tuning_step:{model.tuning_step:.6f} ')
        train(model, train_loader, optimizer, criterion)
        if epoch % epoch_per_test == 0:
            train_acc1, train_acc2, train_acc3 = test(model, train_loader)
            test_acc1, test_acc2, test_acc3 = test(model, test_loader)
            if epoch % print_interval == 0:
                print(f'learning_rate:{rate_rel:.6f},{rate_root:.6f} ')
                print(f'Train Acc with criteria {correct_standard_1:.3f}: {train_acc1:.6f}, Test Acc: {test_acc1:.6f}')
                print(f'Train Acc with criteria {correct_standard_2:.3f}: {train_acc2:.6f}, Test Acc: {test_acc2:.6f}')
                print(f'Train Acc with criteria {correct_standard_3:.3f}: {train_acc3:.6f}, Test Acc: {test_acc3:.6f}')

        if print_param and epoch % epoch_per_print == 0:
            for i in model.named_parameters():
                name, parameters = i
                file = 'Analyze\\' + str(k) + '-' + str(name) + '.txt'
                # f = open(file, 'w')
                if GPU:
                    parameters = parameters.cpu().data.numpy()
                else:
                    parameters = parameters.data.numpy()
                parameters = np.round(parameters, decimals=10)

                # print(parameters)

                np.savetxt(file, parameters, fmt="%.10f")
                '''for row in parameters:
                    for col in row:
                        f.write(str(col))
                        f.write(',')
                    f.write('\n')'''
                # f.close()
                # torch.save(model, 'model.pkl')
                # torch.save(model.state_dict(), 'model_params.pkl')


if __name__ == '__main__':
    main()
