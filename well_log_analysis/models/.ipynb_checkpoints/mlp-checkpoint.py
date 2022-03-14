import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_features=10, hidden_layers=[64, 64, 64], output_features=20, sequence_length=1):
        super().__init__()
        layers = [input_features * sequence_length] + hidden_layers + [output_features]

        self.fc_layers = nn.ModuleList()

        self.activation = nn.SELU()

        for i in range(1, len(layers)):
            fc = nn.Linear(layers[i - 1], layers[i])
            self.fc_layers.append(fc)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.fc_layers[-1](x)
        return x
