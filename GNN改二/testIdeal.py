import Ideal
from Ideal import rel, root
import torch
from config_test import *
import random
import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, BatchNorm2d
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
from GetDataset import get_dataset
import numpy as np
from Gnn_test import GConv

num_epoch = 1  # 训练轮数


def test(model, test_loader, criterion):
    model.eval()

    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    # data_no=0
    avg_loss = 0
    for data in test_loader:
        if GPU:
            data = data.cuda()
        out = model(data.x, data.edge_index, data.batch)
        out = out.reshape(-1)
        loss = criterion(out, data.y)
        avg_loss += loss.item()

        # if data_no==0:
        #     print(out)
        #     print(data.y)
        #     data_no=1

        num_v = torch.numel(out)
        for i in range(num_v):
            total += 1
            if True:
                if abs(out[i] - data.y[i]) < correct_standard_1:
                    correct1 += 1
                if abs(out[i] - data.y[i]) < correct_standard_2:
                    correct2 += 1
                if abs(out[i] - data.y[i]) < correct_standard_3:
                    correct3 += 1

                    # print(out[i],data.y[i],abs(out[i] - data.y[i]) < 0.5)
    avg_loss = avg_loss / len(test_loader)
    print('\033[0;31m' + f'loss: {avg_loss:.10f}' + '\033[0m')

    return correct1 / total, correct2 / total, correct3 / total


def main():
    train_dataset, test_dataset = get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GConv()
    model.conv1.lin_rel.weight.data = torch.FloatTensor(rel())
    model.conv1.lin_root.weight.data = torch.FloatTensor(root())

    criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.MSELoss()
    for epoch in range(1, num_epoch + 1):
        test_acc1, test_acc2, test_acc3 = test(model, test_loader, criterion)
        if epoch % print_interval == 0:
            print(f'Test Acc with criteria {correct_standard_1:.3f}: {test_acc1:.6f}')
            print(f'Test Acc with criteria {correct_standard_2:.3f}: {test_acc2:.6f}')
            print(f'Test Acc with criteria {correct_standard_3:.3f}: {test_acc3:.6f}')


if __name__ == '__main__':
    main()
