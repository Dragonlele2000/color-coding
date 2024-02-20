# 从文件获取数据集
from config_test import *
from VisualizeGraph import visualize_graph
import random
import torch
from torch_geometric.data import Data


def get_dataset_from_file():
    file = open(file_name, 'r')
    n = int(file.readline())
    #修改
    n = int(n * use_ratio)
    # if n != num_instance:
    #     print("Num_instance not consistent!")
    dataset = []
    for instance in range(n):

        v, positive = file.readline().split(' ')
        v, positive = int(v), int(positive)
        e_i = [int(i) for i in file.readline().split(' ')]
        e_j = [int(j) for j in file.readline().split(' ')]
        v_color = [int(c) for c in file.readline().split(' ')]
        y = [int(y_i) for y_i in file.readline().split(' ')]
        # 改动改动改动改动改动改动改动改动改动改动改动改动
        if change_cri or use_max:
            for i in range(len(y)):
                if y[i] > 1:
                    y[i] = 1

        x = [[] for _ in range(v)]
        for v_i in range(v):
            color = v_color[v_i]
            x[v_i] = [0 for _ in range(k * (1 << k))]
            x[v_i][color * (1 << k) + (1 << color)] = 1
        edge_index = torch.tensor([e_i, e_j], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)

    file.close()
    return dataset


def get_dataset():
    dataset = get_dataset_from_file()
    if draw_data:
        color = []
        for v_x in dataset[data_id].x:
            for i in range(k):
                temp = i * (1 << k) + (1 << i)
                if v_x[temp] == 1:
                    color.append(i)
                    break

        visualize_graph(dataset[data_id], color)
    torch.manual_seed(randomSeed)
    random.seed(randomSeed)
    random.shuffle(dataset)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)

    dataset = dataset[:total_size]
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    return train_dataset, test_dataset
