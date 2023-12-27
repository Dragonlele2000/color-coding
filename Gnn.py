# GNN主体
from config import *
import random
import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
from GetDataset import get_dataset


class GConv(torch.nn.Module):
    def __init__(self):
        super(GConv, self).__init__()
        torch.manual_seed(randomSeed)

        self.conv1 = GraphConv(num_node_features, num_node_features)
        self.lin = Linear(num_node_features, 1, bias=False)

    def forward(self, x, edge_index, batch):
        for i in range(k):
            x = self.conv1(x, edge_index)
            x = x.relu()
        x = self.lin(x)

        # print("forward result:",x)
        return x


count = 0


def train(model, train_loader, optimizer, criterion):
    model.train()
    global count
    for data in train_loader:
        # data=data.cuda()
        count += 1
        out = model(data.x, data.edge_index, data.batch)
        out = out.reshape(-1)
        # print(out)
        # print(data.y)
        loss = criterion(out, data.y)
        # print(loss)
        # print(data.x)
        if count % print_interval == 0:
            print(f'loss: {loss.item():04f}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, test_loader):
    model.eval()

    correct = 0
    total = 0
    # data_no=0
    for data in test_loader:
        # data=data.cuda()
        out = model(data.x, data.edge_index, data.batch)
        out = out.reshape(-1)

        # if data_no==0:
        #     print(out)
        #     print(data.y)
        #     data_no=1

        num_v = torch.numel(out)
        for i in range(num_v):
            total += 1
            if abs(out[i] - data.y[i]) < 0.5:
                correct += 1
        # print(out)
        # print(data.y)
        # correct += int((pred == data.y).sum())
    return correct / total


def main():
    # print(f"GPU: {torch.cuda.device_count():01d}")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    train_dataset, test_dataset = get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train_loader = train_dataset
    # test_loader  test_dataset
    model = GConv().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.CrossEntropyLoss()
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = torch.nn.L1Loss()
    criterion=torch.nn.MSELoss()

    for epoch in range(1, num_epoch + 1):
        if epoch % print_interval == 0:
            print('-'*50)
            print(f'Epoch: {epoch:03d}')
        train(model, train_loader, optimizer, criterion)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        if epoch % print_interval == 0:
            ExpLR.step()
            rate = optimizer.param_groups[0]['lr']
            print(f'learning_rate:{rate:04f} ')
            print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


main()
