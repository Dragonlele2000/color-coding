# 根据配置生成或从文件获取数据集
from config import *
from GenerateDataset import generate_dataset
from GetDatasetFromFile import get_dataset_from_file
from VisualizeGraph import visualize_graph
import random
import torch


def get_dataset():
    dataset = []
    if inputFile:
        dataset = generate_dataset()
    if readFromFile:
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
    random.shuffle(dataset)
    train_dataset = dataset[:train_n]
    test_dataset = dataset[train_n:]
    return train_dataset, test_dataset
