import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Neural network model for binary classification.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize the model.

        Args:
            input_size (int): The input size of the model.
        """
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim // 2)
        self.fc2 = nn.Linear(1, output_dim // 2)

    def forward(self, x, RA):
        """
        Forward pass of the model.

        Args:
            x (tensor): The input tensor.
            RA (tensor): The RA value.

        Returns:
            tensor: The output tensor.
        """
        x = self.fc1(x)
        ra = self.fc2(RA)
        return torch.cat((x, ra), dim=1)

class MODEL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MODEL, self).__init__()
        self.emb = FeatureExtractor(input_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, RA):
        return self.fc(self.emb(x, RA))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device = 'cpu'):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out