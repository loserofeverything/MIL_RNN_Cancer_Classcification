import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

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
    
class GAttNet(nn.Module):
    def __init__(self, input_dim, instance_hidden_dim, instance_concepts, bag_hidden_dim, num_class):
        super(GAttNet, self).__init__()
        self.W = nn.Parameter(torch.Tensor(instance_hidden_dim, instance_concepts))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.embs = FeatureExtractor(input_dim, instance_hidden_dim)
        self.V = nn.Parameter(torch.Tensor(instance_hidden_dim, bag_hidden_dim))
        init.kaiming_uniform_(self.V, a=math.sqrt(5))
        self.wB = nn.Parameter(torch.Tensor(bag_hidden_dim, 1))
        init.kaiming_uniform_(self.wB, a=math.sqrt(5))
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bag_hidden_dim, num_class)
            )

    def forward(self, x, ra):
        fx = self.embs(x, ra)
        HI = torch.mm(fx.t(), F.softmax(torch.mm(fx, self.W), dim =1))
        HB = torch.mm(self.V.t(), HI)
        b = torch.mm(HB, F.softmax(torch.mm(HB.t(), self.wB), dim = 1)).t()
        outputs = self.fc(b)
        return outputs, HI


# random_tensor = torch.randn(100, 20)
# random_ra = torch.randn(100, 1)
# gattnet = GAttNet(20, 128, 32, 64,7)
# output = gattnet(random_tensor, random_ra)
# print(output.size())