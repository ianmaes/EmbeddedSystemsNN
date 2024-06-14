from torchviz import make_dot
from NN import NN
import torch
import pandas as pd
import sklearn.model_selection


model = NN(7, 64, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()

data = pd.read_csv('SensorData.csv')
X = data.iloc[:, :7].values
y = data.iloc[:, -1].values / 100
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.15, random_state=41)

X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.unsqueeze(torch.tensor(y_train, dtype=torch.float32), 1), torch.unsqueeze(torch.tensor(y_test, dtype=torch.float32), 1)

m = X_train.mean(0, keepdim=True)
s = X_train.std(0, unbiased=False, keepdim=True)

X_train -= m

y_hat = model(X_train[0])
make_dot(y_hat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
