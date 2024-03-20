# 画图
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def visualize_graph(data, color):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    plt.show()


if __name__ == "__main__":
    e_i = [0, 1, 2, 3]
    e_j = [1, 2, 3, 0]
    x = [[0, 0], [1, 1], [2, 2], [1, 1]]
    y = [0, 1, 2, 1]

    edge_index = torch.tensor([e_i, e_j], dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    visualize_graph(data, color=y)
