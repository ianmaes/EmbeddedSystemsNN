import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from NN import NN

data = pd.read_csv('data/iris.csv')


epochs = 1000
lr = 0.01
optimizer = optim.Adam(NN.parameters(), lr=lr)

for i in range(epochs):
    optimizer.zero_grad()
    x = torch.tensor(data.iloc[:, :4].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, 4].values, dtype=torch.int64)
    y_pred = NN(x)
    loss = nn.CrossEntropyLoss()(y_pred, y)
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {}'.format(i, loss.item()))

# torch.save(NN.state_dict(), 'model.pth')

