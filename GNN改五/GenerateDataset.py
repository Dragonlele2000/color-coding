# 生成数据集
import random
import torch
from Gnn_test import GConv
from Ideal import rel, root
from VisualizeGraph import visualize_graph
from Solve import solve
from torch_geometric.data import Data
from config_generate import *
GPU=False


def create_graph(v_l, v_u, k,model):
    v = random.randint(v_l, v_u)
    allColor = random.sample(range(v), k)
    e_i = []
    e_j = []
    v_color = []
    x = [[] for _ in range(v)]
    for v_i in range(v):
        if v_i in allColor:
            color = allColor.index(v_i)
        else:
            color = random.randint(0, k - 1)
        v_color.append(color)
        x[v_i] = [0 for _ in range(k * (1 << k))]
        x[v_i][color * (1 << k) + (1 << color)] = 1
        x[v_i][color * (1 << k)] = 1
        for v_j in range(v_i + 1, v):
            if random.random() < p:
                e_i.append(v_i)
                e_i.append(v_j)
                e_j.append(v_j)
                e_j.append(v_i)
    edge_index = torch.tensor([e_i, e_j], dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    if GPU:
        x=x.cuda()
        edge_index=edge_index.cuda()

    y, positive = solve(x, edge_index,model)

    y = y.int()
    data = Data(x=x, edge_index=edge_index, y=y)

    success = True
    if (y > upper_bound).sum() > 0:
        success = False
    return success, data, v, e_i, e_j, v_color, y, positive


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
        f.write(str(i.item()) + " ")
    # print(y[-1])
    f.write(str(y[-1].item()) + "\n")


def generate_dataset():
    n = num_instance
    dataset = []
    random.seed(randomSeed)
    num_1=0
    total=0
    maximum_path=0

    model = GConv()
    model.e = torch.zeros((1, (1<<k)*k))
    if GPU:
        model.e = model.e.cuda()
    for i in range(k):
        model.e[0][(i + 1) * (1 << k) - 1] = 1
    model.conv1.lin_rel.weight.data = torch.FloatTensor(rel(k))
    model.conv1.lin_root.weight.data = torch.FloatTensor(root(k))
    if GPU:
        model = model.cuda()
    f = open(file_name, "w")
    f.write(str(n)+' '+str(upper_bound))
    f.write("\n")
    count = 0
    for instance in range(n):
        if instance % print_epoch == 0:
            print("Generating", instance, "-", instance + print_epoch)
        while True:
            success, data, v, e_i, e_j, v_color, y, positive = create_graph(v_l, v_u, k,model)
            if success and len(e_i) >= lower_bound:
                dataset.append(data)
                num_1+=(y>=1).sum().item()
                total+=y.numel()
                maximum_path=max(maximum_path,y.max().item())
                break
            elif show_failure:
                count += 1
                if count % print_epoch == 0:
                    print('Failure No.', count)
        write_to_file(v, e_i, e_j, v_color, y, positive, f)
        if instance % print_epoch == 0:
            print(f"has path ratio = {num_1 / total:.05f}")

    f.close()
    return dataset


def main():
    appr = input("Continue with k = " + str(k) + '(y or n)\n')
    if appr == 'y':
        generate_dataset()
    else:
        print('denied')
        exit(1)


if __name__ == '__main__':
    main()
