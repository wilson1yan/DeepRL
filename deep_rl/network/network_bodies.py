#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class BaseBody(nn.Module):
    def __init__(self):
        super(BaseBody, self).__init__()

    def forward(self, x):
        return x

    def reset(self):
        pass

class LSTMConvBody(BaseBody):
    def __init__(self, warmup, seq_len,
                 hidden_size=256, num_layers=1, in_channels=1):
        super(LSTMConvBody, self).__init__()
        self.warmup = warmup
        self.seq_len = seq_len
        assert self.warmup < self.seq_len

        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.lstm = nn.LSTM(input_size=7 * 7 * 64, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.feature_dim = hidden_size

        self.start_episode = True

    def forward(self, x):
        if self.training:
            batch_size = x.size(0) // self.seq_len

            y = F.relu(self.conv1(x))
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y))
            y = y.view(batch_size, self.seq_len, -1)

            y1, y2 = y[:, :self.warmup, :], y[:, self.warmup:, :]

            _, (h, c) = self.lstm(y1)
            h.detach(); c.detach()
            y, _ = self.lstm(y2, (h, c))
            y = y.contiguous()[:, -1]
        else:
            y = F.relu(self.conv1(x))
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y))
            y = y.view(y.size(0), -1)
            y = y.unsqueeze(1)
            if self.start_episode:
                y, (self.h, self.c) = self.lstm(y)
                self.start_episode = False
            else:
                y, (self.h, self.c) = self.lstm(y, (self.h, self.c))
            y = y.squeeze(1)
        return y

    def reset(self):
        self.start_episode = True

class NatureConvBody(BaseBody):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class DDPGConvBody(BaseBody):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(BaseBody):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class TwoLayerFCBodyWithAction(BaseBody):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(BaseBody):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(BaseBody):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
