import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sklearn
from NN import NN


data = pd.read_csv('SensorData_final.csv')

X = data.iloc[:, :5].values
y = data.iloc[:, -1].values
print(y)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.15, random_state=41)

X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.unsqueeze(torch.tensor(y_train, dtype=torch.float32), 1), torch.unsqueeze(torch.tensor(y_test, dtype=torch.float32), 1)

m = X_train.mean(0, keepdim=True)
s = X_train.std(0, unbiased=False, keepdim=True)

X_train -= m
X_train /= s
X_test -= m

X_test /= s


if __name__ == '__main__':
    

    criterion = nn.MSELoss()
    model = NN(5, 64, 1)
    epochs = 7001
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(i, loss.item()))
    print(y_pred*100)
    print(y_train*100)

    torch.save(model.state_dict(), 'model1.pth')
