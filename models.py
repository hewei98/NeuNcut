import torch
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    
    def __init__(self, input_dims, hid_dims, out_dims):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)

        # initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        return h