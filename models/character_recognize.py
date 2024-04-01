import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 26)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, input: torch.Tensor, train=False):
        input = input.view(-1, 1, 28, 28)
        # size = X * 28*28 * 1

        output = self.relu(self.pool(self.conv1(input)))
        # size = X * 12*12 * 6
        output = self.relu(self.pool(self.conv2(output)))
        # size = X * 4*4 * 16

        output = output.view(-1, 16 * 4 * 4)  # flattening
        # size = X * 256

        if train: output = self.dropout(output)
        output = self.l1(output)
        output = self.relu(output)

        if train: output = self.dropout(output)
        output = self.l2(output)
        output = self.relu(output)

        if train: output = self.dropout(output)
        output = self.l3(output)
        output = self.sig(output)
        return output


class CharacterRecognize:
    def __init__(self):
        None


if __name__ == '__main__':
    None
