import math

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import functional as F


class EMLPBaseline(nn.Module):
    def __init__(self, input_dimension):
        super(EMLPBaseline, self).__init__()
        self.input_dimension = input_dimension
        hidden_units = int(max(30, round(input_dimension / 30)) * 10)
        out_units = 1

        self.fc1 = nn.Linear(input_dimension, hidden_units)
        self.dropout1 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(p=0.25)

        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.dropout3 = nn.Dropout(p=0.25)

        self.fc4 = nn.Linear(hidden_units, hidden_units)
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc5 = nn.Linear(hidden_units, out_units)

    def forward(self, x):
        h1 = F.relu(self.dropout1(self.fc1(x)))
        h2 = F.relu(self.dropout2(self.fc2(h1)))
        h3 = F.relu(self.dropout3(self.fc3(h2)))
        h4 = F.relu(self.dropout4(self.fc4(h3)))
        out = torch.sigmoid(self.fc5(h4))
        return out


class EMLPSimplified(nn.Module):
    def __init__(self, input_dimension):
        super(EMLPSimplified, self).__init__()
        self.input_dimension = input_dimension
        hidden_units = int(max(30, round(input_dimension / 30)) * 10)
        out_units = 1
        self.fc1 = nn.Linear(input_dimension, hidden_units)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(hidden_units, out_units)

    def forward(self, x):
        h1 = F.relu(self.dropout1(self.fc1(x)))
        out = torch.sigmoid(self.fc2(h1))
        return out


def MLP(features, label, random_seed, model):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    input_dimension = len(features.columns)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emlp = model(input_dimension).to(device)
    loss_fn = nn.MSELoss()
    lr = 1e-2
    min_batch = 40
    batch_size = min_batch if len(label) >= min_batch else len(label)
    optim = torch.optim.SGD(emlp.parameters(), lr=lr, momentum=0.9)

    torch_dataset = Data.TensorDataset(torch.tensor(features.values, dtype=torch.float32),
                                       torch.tensor(label.values, dtype=torch.float32))
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             )

    EPOCH = 200
    for epoch in range(1, EPOCH + 1):
        emlp.train()
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optim.zero_grad()
            out = emlp(batch_x)
            loss = loss_fn(out.squeeze(-1), batch_y)
            loss.backward()
            train_loss += float(loss.item())
            optim.step()
        if epoch % 20 == 0:
            print('====>MLP training... Epoch: {} total loss: {:.4f}'.format(epoch, train_loss))

    emlp.eval()
    ret_dict = {}
    with torch.no_grad():
        virtual_test = torch.eye(input_dimension).to(device)
        suspicious = emlp(virtual_test)
        for line, s in zip(features.columns, suspicious):
            ret_dict[line] = s.item()
    return ret_dict


def MLPBaseline(features, label, random_seed):
    return MLP(features, label, random_seed, EMLPBaseline)


def MLPSimplified(features, label, random_seed):
    return MLP(features, label, random_seed, EMLPSimplified)


class ECNN(nn.Module):
    def __init__(self, input_dimension, mid_channels, out_channels):
        super(ECNN, self).__init__()

        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.step = 2

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.mid_channels,
                               kernel_size=(1, self.kernel_size))

        self.conv2 = nn.Conv2d(in_channels=self.mid_channels,
                               out_channels=self.out_channels,
                               kernel_size=(1, self.kernel_size))
        self.hidden_units = math.floor((input_dimension - self.kernel_size + 1) / self.step)
        self.hidden_units = self.out_channels * (math.floor((self.hidden_units - self.kernel_size + 1) / self.step))
        self.hidden_units1 = int(math.floor(math.sqrt(self.hidden_units)))
        self.fc1 = nn.Linear(self.hidden_units, self.hidden_units1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(self.hidden_units1, 1)

    def forward(self, x):
        h1 = F.max_pool2d(F.relu(self.conv1(x)), (1, self.step))
        h2 = F.max_pool2d(F.relu(self.conv2(h1)), (1, self.step))
        h2 = h2.view(-1, self.hidden_units)
        h3 = F.relu(self.dropout1(self.fc1(h2)))
        out = torch.sigmoid(self.fc2(h3))
        return out


class ECNNBaseline(ECNN):
    def __init__(self, input_dimension):
        super(ECNNBaseline, self).__init__(input_dimension, 15, 30)


class ECNNSimplified(ECNN):
    def __init__(self, input_dimension):
        super(ECNNSimplified, self).__init__(input_dimension, 10, 20)


def CNN(features, label, random_seed, model):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    input_dimension = len(features.columns)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ecnn = model(input_dimension).to(device)
    lr = 0.01
    optim = torch.optim.SGD(ecnn.parameters(), lr=lr, momentum=0.9)
    min_batch = 10
    batch_size = min_batch if len(label) >= min_batch else len(label)
    loss_fn = nn.MSELoss()

    torch_dataset = Data.TensorDataset(torch.tensor(features.values, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                       torch.tensor(label.values, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             )

    EPOCH = 200
    for epoch in range(EPOCH + 1):
        ecnn.train()
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optim.zero_grad()
            out = ecnn(batch_x)
            loss = loss_fn(out, batch_y.view(-1, 1))
            loss.backward()
            train_loss += float(loss.item())
            optim.step()
        if epoch % 20 == 0:
            print('====>CNN training... Epoch: {}  total loss: {:.4f}'.format(epoch, train_loss))

    ecnn.eval()
    ret_dict = {}
    with torch.no_grad():
        virtual_test = torch.eye(input_dimension).unsqueeze(0).unsqueeze(0).to(device)
        suspicious = ecnn(virtual_test)
        for line, s in zip(features.columns, suspicious):
            ret_dict[line] = s.item()
    return ret_dict


def CNNBaseline(features, label, random_seed):
    return CNN(features, label, random_seed, ECNNBaseline)


def CNNSimplified(features, label, random_seed):
    return CNN(features, label, random_seed, ECNNSimplified)


class ERNN(nn.Module):
    def __init__(self, num_in, num_layers):
        super(ERNN, self).__init__()
        n_hidden = num_in
        self.feature_extraction = nn.Sequential(
            nn.Linear(in_features=num_in, out_features=n_hidden, bias=True),
            nn.RNN(
                input_size=n_hidden,
                hidden_size=n_hidden,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=n_hidden * 2, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extraction(x)[0]
        x = self.classifier(x)
        return x


class ERNNBaseline(ERNN):
    def __init__(self, num_in):
        super(ERNNBaseline, self).__init__(num_in, 2)


class ERNNSimplified(ERNN):
    def __init__(self, num_in):
        super(ERNNSimplified, self).__init__(num_in, 1)


def RNN(features, label, random_seed, model):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    input_dimension = len(features.columns)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ernn = model(input_dimension).to(device)
    loss_fn = nn.MSELoss()
    lr = 1e-2
    min_batch = 40
    batch_size = min_batch if len(label) >= min_batch else len(label)
    optim = torch.optim.SGD(ernn.parameters(), lr=lr, momentum=0.9)

    torch_dataset = Data.TensorDataset(torch.tensor(features.values, dtype=torch.float32).unsqueeze(0),
                                       torch.tensor(label.values, dtype=torch.float32).unsqueeze(0))
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             )

    EPOCH = 200
    for epoch in range(1, EPOCH + 1):
        ernn.train()
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optim.zero_grad()
            out = ernn(batch_x)
            loss = loss_fn(out.squeeze(-1), batch_y)
            loss.backward()
            train_loss += float(loss.item())
            optim.step()
        if epoch % 20 == 0:
            print('====>RNN training... Epoch: {} total loss: {:.4f}'.format(epoch, train_loss))

    ernn.eval()
    ret_dict = {}
    with torch.no_grad():
        virtual_test = torch.eye(input_dimension).unsqueeze(0).to(device)
        suspicious = ernn(virtual_test).squeeze(0)
        for line, s in zip(features.columns, suspicious):
            ret_dict[line] = s.item()
    return ret_dict


def RNNBaseline(features, label, random_seed):
    return RNN(features, label, random_seed, ERNNBaseline)


def RNNSimplified(features, label, random_seed):
    return RNN(features, label, random_seed, ERNNSimplified)
