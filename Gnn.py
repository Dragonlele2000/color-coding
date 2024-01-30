# GNN主体
from config import *
import random
import torch
from torch.nn import Linear, BatchNorm1d, BatchNorm2d
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
from GetDataset import get_dataset
import numpy as np


class GConv(torch.nn.Module):
    def __init__(self):
        super(GConv, self).__init__()
        torch.manual_seed(randomSeed)

        if use_max:
            self.conv1 = GraphConv(num_node_features, num_node_features, str='max', bias=False)
        else:
            self.conv1 = GraphConv(num_node_features, num_node_features, str='add', bias=False)
        self.lin = Linear(num_node_features, 1, bias=False)
        self.m = BatchNorm1d(num_node_features)

    def forward(self, x, edge_index, batch):
        for i in range(k):
            x = self.conv1(x, edge_index)
            # x = self.m(x)
            x = x.relu()
        # x = self.lin(x)
        x = x[:, (1 << k) - 1::(1 << k)]
        x = torch.sum(x, dim=1)
        # print("forward result:",x)
        # return x
        return x


count = 0


def train(model, train_loader, optimizer, criterion):
    model.train()
    global count
    count += 1
    avg_loss = 0
    for data in train_loader:
        if GPU:
            data = data.cuda()
        out = model(data.x, data.edge_index, data.batch)
        out = out.reshape(-1)

        # print(out)
        # print(data.y)
        # print(len(out))
        # s=0
        # for i in range(len(out)):
        #     s+=(out[i]-data.y[i])**2
        # print(s/len(out))

        # print(out)
        # print(data.y)
        loss = criterion(out, data.y)
        avg_loss += loss.item()
        # print(loss)
        # print(data.x)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if count % print_interval == 0:
        print(f'loss: {avg_loss/len(train_loader):04f}')


def test(model, test_loader):
    model.eval()

    correct = 0
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
                if abs(out[i] - data.y[i]) < 0.5:
                    correct += 1
                    # print(out[i],data.y[i],abs(out[i] - data.y[i]) < 0.5)

        # print(out)
        # print(data.y)
        # print(correct, total)
    return correct / total


def main():
    # print(f"GPU: {torch.cuda.device_count():01d}")
    if GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    train_dataset, test_dataset = get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train_loader = train_dataset
    # test_loader  test_dataset
    model = GConv()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.CrossEntropyLoss()
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    if GPU:
        model = model.cuda()
        criterion = criterion.cuda()

    rate = learning_rate
    for epoch in range(1, num_epoch + 1):
        if epoch % print_interval == 0:
            print('-' * 50)
            print(f'Epoch: {epoch:03d}')
        train(model, train_loader, optimizer, criterion)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        if epoch % print_interval == 0:
            if gamma != 1:
                ExpLR.step()
                rate = optimizer.param_groups[0]['lr']
            print(f'learning_rate:{rate:04f} ')
            print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    if print_param:
        f = 0
        for i in model.named_parameters():
            name, parameters = i
            file = 'Analyze\\' + str(name) + '.txt'
            f = open(file, 'w')
            parameters = parameters.data.numpy()
            print(parameters)
            for row in parameters:
                for col in row:
                    f.write(str(col))
                    f.write(',')
                f.write('\n')
        f.close()


main()
