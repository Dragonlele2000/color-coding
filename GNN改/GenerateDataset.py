# 生成数据集
import random
import torch
from VisualizeGraph import visualize_graph
from Solve import solve
from torch_geometric.data import Data
from config_generate import *


def create_graph(v_l, v_u, k):
    v = random.randint(v_l, v_u)
    allColor = random.sample(range(v), k)
    e_i = []
    e_j = []
    v_color = []
    x = [[] for _ in range(v)]
    e = [[0 for _ in range(v)] for _ in range(v)]
    for v_i in range(v):
        if v_i in allColor:
            color = allColor.index(v_i)
        else:
            color = random.randint(0, k - 1)
        v_color.append(color)
        x[v_i] = [0 for _ in range(k * (1 << k))]
        x[v_i][color * (1 << k) + (1 << color)] = 1

        for v_j in range(v_i + 1, v):
            if random.random() < p:
                e_i.append(v_i)
                e_i.append(v_j)
                e_j.append(v_j)
                e_j.append(v_i)
                e[v_i][v_j] = 1
                e[v_j][v_i] = 1
    edge_index = torch.tensor([e_i, e_j], dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)

    y, positive = solve(v, v_color, e, k)

    tensor_y = torch.tensor(y, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=tensor_y)

    if debug:
        print("v=", v)
        print("color=", end='')
        for i in v_color:
            print(i, end=" ")
        print()
        for i in e_i:
            print(i, end=' ')
        print()
        for i in e_j:
            print(i, end=' ')
    if debug2:
        print(y)
        data = Data(x=x, edge_index=edge_index)
        visualize_graph(data, color=v_color)

    return data, v, e_i, e_j, v_color, y, positive


def write_to_file(v, e_i, e_j, v_color, y, positive, f):
    # print(v,positive)
    f.write(str(v) + " " + str(positive) + "\n")
    for i in e_i[:-1]:
        # print(i,end=' ')
        f.write(str(i) + " ")
    # print(e_i[-1])
    f.write(str(e_i[-1]) + "\n")
    for i in e_j[:-1]:
        # print(i,end=' ')
        f.write(str(i) + " ")
    # print(e_j[-1])
    f.write(str(e_j[-1]) + "\n")
    for i in v_color[:-1]:
        # print(i,end=' ')
        f.write(str(i) + " ")
    # print(v_color[-1])
    f.write(str(v_color[-1]) + "\n")
    for i in y[:-1]:
        # print(i, end=' ')
        f.write(str(i) + " ")
    # print(y[-1])
    f.write(str(y[-1]) + "\n")


def generate_dataset():
    n = num_instance
    dataset = []
    random.seed(randomSeed)

    f = open(file_name, "w")
    f.write(str(n))
    f.write("\n")
    for instance in range(n):
        if instance % 5000 == 0:
            print("Generating", instance, "-", instance + 5000)
        while True:
            data, v, e_i, e_j, v_color, y, positive = create_graph(v_l, v_u, k)
            if len(e_i) >= k * 2:
                dataset.append(data)
                break
        write_to_file(v, e_i, e_j, v_color, y, positive, f)
    f.close()
    return dataset


def main():
    generate_dataset()


if __name__ == '__main__':
    main()
