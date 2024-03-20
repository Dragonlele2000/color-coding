# GNN主体
import math

from Ideal import root, get_upper_bound
from config_test import *
import random
import torch
from torch.nn import Linear, BatchNorm1d, BatchNorm2d
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
from GetDataset import get_dataset
import torch.nn as nn
import numpy as np
import torch_scatter

class GConv(torch.nn.Module):
    def __init__(self):
        super(GConv, self).__init__()
        torch.manual_seed(randomSeed)
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
        # torch.nn.init.normal_(self.conv1.lin_rel.weight, 0, 0.05)
        # torch.nn.init.normal_(self.conv1.lin_root.weight, -1, 1)
        # torch.nn.init.kaiming_uniform_(self.conv1.lin_rel.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_uniform_(self.conv1.lin_root.weight, nonlinearity='relu')
        if use_elu:
            self.elu = nn.ELU(alpha=alpha)
        if use_leaky_relu:
            self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        if use_celu:
            self.celu = nn.CELU(alpha=alpha)
        if use_tanh:
            self.tanh = nn.Tanh()
        if fix_root:
            self.conv1.lin_root.weight.data = torch.FloatTensor(root())
            self.conv1.lin_root.requires_grad_(False)
        # print(self.conv1.lin_root.weight)

    def forward(self, x, edge_index, batch):
        # print(self.conv1.lin_rel.weight)
        # print(x)
        # print(self.conv1.lin_rel.weight)
        for i in range(k - 1):
            if use_avg:
                x = x / avg_ratio
            x = self.conv1(x, edge_index)
            # x = self.m(x)
            if use_elu:
                x = self.elu(x)
            elif use_leaky_relu:
                x = self.leaky_relu(x)
            elif use_celu:
                x = self.celu(x)
            elif use_tanh:
                x = self.tanh(x)
            else:
                x = x.relu()
        # x = self.lin(x)
        if use_linear:
            x = self.linear(x)
        else:
            x = (torch.matmul(self.e, x.T)).T
        if use_avg:
            x = (avg_ratio ** (k - 1)) * x

        return x


count = 0


def train(model, train_loader, optimizer, criterion):
    model.train()
    global count
    count += 1
    avg_loss = 0
    train_count = 0
    for data in train_loader:
        train_count += 1
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
        print('\033[0;31m' + f'loss: {avg_loss:.10f}' + '\033[0m')
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

        # if data_no==0:
        #     print(out)
        #     print(data.y)
        #     data_no=1

        num_v = torch.numel(out)
        for i in range(num_v):
            total += 1
            # 改动改动改动改动改动改动改动改动改动改动改动改动
            # if change_cri:
            #     if out[i] > 0.5 and data.y[i] == 1:
            #         correct += 1
            #     elif out[i] <= 0.5 and data.y[i] == 0:
            #         correct += 1
            # else:
            if True:
                if abs(out[i] - data.y[i]) < correct_standard_1:
                    correct1 += 1
                if abs(out[i] - data.y[i]) < correct_standard_2:
                    correct2 += 1
                if abs(out[i] - data.y[i]) < correct_standard_3:
                    correct3 += 1

                    # print(out[i],data.y[i],abs(out[i] - data.y[i]) < 0.5)

        # print(out)
        # print(data.y)
        # print(correct, total)
    return correct1 / total, correct2 / total, correct3 / total


class my_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow(x - y, 2))


def main():
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.CrossEntropyLoss()
    if use_ExpLR:
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.SmoothL1Loss()
    if GPU:
        model = model.cuda()
        criterion = criterion.cuda()

    rate = learning_rate
    count_down = 0
    for epoch in range(1, num_epoch + 1):
        if count_down == 0:
            hasException = True
            new_rate = -1
            while hasException:
                try:
                    pack = input(f'输入训练次数 学习率(当前{rate:.6f})：').split(' ')
                    if len(pack) == 2:
                        count_down = int(float(pack[0]))
                        new_rate = float(pack[1])
                        hasException = False
                        rate = new_rate
                        optimizer.param_groups[0]['lr'] = rate
                    elif len(pack) == 1:
                        count_down = int(pack[0])
                        hasException = False
                except Exception:
                    print('输入格式错误')
                    hasException = True
        count_down -= 1
        if epoch % print_interval == 0:
            print('-' * 50)
            print(f'Epoch: {epoch:03d}')
            print(f'learning_rate:{rate:.6f} ')
        lr_loss = train(model, train_loader, optimizer, criterion)
        if epoch % epoch_per_test == 0:
            train_acc1, train_acc2, train_acc3 = test(model, train_loader)
            test_acc1, test_acc2, test_acc3 = test(model, test_loader)
            if epoch % print_interval == 0:
                print(f'learning_rate:{rate:.6f} ')
                print(f'Train Acc with criteria {correct_standard_1:.3f}: {train_acc1:.6f}, Test Acc: {test_acc1:.6f}')
                print(f'Train Acc with criteria {correct_standard_2:.3f}: {train_acc2:.6f}, Test Acc: {test_acc2:.6f}')
                print(f'Train Acc with criteria {correct_standard_3:.3f}: {train_acc3:.6f}, Test Acc: {test_acc3:.6f}')

        if use_lr_loss and lr_loss * lr_loss_ratio < max_loss_rate:
            optimizer.param_groups[0]['lr'] = lr_loss * lr_loss_ratio
            rate = lr_loss * lr_loss_ratio
        elif use_ExpLR and epoch % epoch_change_lr == 0 and gamma != 1 and rate != min_rate:
            if rate * gamma < min_rate:
                optimizer.param_groups[0]['lr'] = min_rate
                rate = min_rate
            else:
                ExpLR.step()
                rate = optimizer.param_groups[0]['lr']
        if print_param and epoch % epoch_per_print == 0:
            for i in model.named_parameters():
                name, parameters = i
                file = 'Analyze\\' + str(k) + '-' + str(name) + '.txt'
                # f = open(file, 'w')
                if GPU:
                    parameters = parameters.cpu().data.numpy()
                else:
                    parameters = parameters.data.numpy()
                parameters = np.round(parameters, decimals=2)

                # print(parameters)

                np.savetxt(file, parameters, fmt="%.2f")
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
