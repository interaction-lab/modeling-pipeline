import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=0,
                dilation=dilation,
            )
        )
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=0,
                dilation=dilation,
            )
        )
        self.net = nn.Sequential(
            self.pad,
            self.conv1,
            self.relu,
            self.dropout,
            self.pad,
            self.conv2,
            self.relu,
            self.dropout,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, num_channels, window, kernel_size=2, num_classes=3, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(
            window, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        return self.decoder(self.dropout(self.tcn(x)[:, :, -1]))


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(dev)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(dev)

        # Forward propagate LSTM
        out, _ = self.rnn(x)
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(dev)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(dev)

        # Forward propagate LSTM
        out, _ = self.rnn(x)
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(dev)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(dev)

        # Forward propagate LSTM
        out, _ = self.rnn(x)
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def test_models():
    # model = RNNModel(input_size=32, hidden_size=3, num_layers=4, num_classes=2, dropout=.25).to(dev)
    model = TCNModel(num_channels=[32] * 2, window=30, kernel_size=3, dropout=0.25)
    # model = LSTMModel(input_size=32, hidden_size=3, num_layers=4, num_classes=2, dropout=.25).to(dev)
    # model = GRUModel(input_size=32, hidden_size=3, num_layers=4, num_classes=3, dropout=.25).to(dev)

    random_data = torch.rand((10, 30, 32)).to(dev)

    model.to(dev)
    result = model(random_data)
    print(result)


if __name__ == "__main__":
    test_models()
