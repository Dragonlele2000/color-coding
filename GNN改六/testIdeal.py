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
use_ratio = .01

def test(model, test_loader, criterion):
    model.eval()

    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    avg_loss = 0
    for data in test_loader:
        if GPU:
            data = data.cuda()
        out = model(data.x, data.edge_index, data.batch)
        out = out.reshape(-1)
        loss = criterion(out, data.y)
        avg_loss += loss.item()

        total += out.numel()
        result = abs(out - data.y)
        correct1 += (result < correct_standard_1).sum().item()
        correct2 += (result < correct_standard_2).sum().item()
        correct3 += (result < correct_standard_3).sum().item()
    avg_loss = avg_loss / len(test_loader)
    print('\033[0;31m' + f'loss: {avg_loss:.10f}' + '\033[0m')
    print(model.conv1.lin_root.weight.data.mean().item())
    return correct1 / total, correct2 / total, correct3 / total


def main():
    print('getting dataset')
    train_dataset, test_dataset = get_dataset()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GConv()
    model.conv1.lin_rel.weight.data = torch.FloatTensor(rel(k))
    model.conv1.lin_root.weight.data = torch.FloatTensor(root(k))
    criterion = torch.nn.SmoothL1Loss()
    if GPU:
        model = model.cuda()
    # criterion = torch.nn.MSELoss()
    print('testing')
    for epoch in range(1, num_epoch + 1):
        test_acc1, test_acc2, test_acc3 = test(model, test_loader, criterion)
        if epoch % print_interval == 0:
            print(f'Test Acc with criteria {correct_standard_1:.3f}: {test_acc1:.6f}')
            print(f'Test Acc with criteria {correct_standard_2:.3f}: {test_acc2:.6f}')
            print(f'Test Acc with criteria {correct_standard_3:.3f}: {test_acc3:.6f}')


if __name__ == '__main__':
    main()
