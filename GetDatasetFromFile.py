# 从文件获取数据集
from config import *
import torch
from torch_geometric.data import Data


def get_dataset_from_file():
    file = open(file_name, 'r')
    n = int(file.readline())
    if n != num_instance:
        print("Num_instance not consistent!")
    dataset = []
    for instance in range(n):

        v, positive = file.readline().split(' ')
        v, positive = int(v), int(positive)
        e_i = [int(i) for i in file.readline().split(' ')]
        e_j = [int(j) for j in file.readline().split(' ')]
        v_color = [int(c) for c in file.readline().split(' ')]
        y = [int(y_i) for y_i in file.readline().split(' ')]
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
