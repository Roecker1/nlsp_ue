import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyWordCNN1d(nn.Module):
    """
    """

    def __init__(self, num_classes, num_features, num_kernels, mem_depth, num_hidden=20):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes to predict
        num_features : int
            Number of signal features per time step
        num_kernels : int
            Number of convolution kernels
        mem_depth : int
            Memory depth = kernel size of the model
        """
        super().__init__()
        # self.num_kernels = num_kernels
        # self.mem_depth = mem_depth
        # self.num_classes = num_classes
        # self.num_features = num_features
        self.conv_layer = nn.Conv1d(num_features, num_kernels, mem_depth, groups=num_features)
        # self.conv_layer = nn.Conv1d(num_features, num_kernels, mem_depth)
        # self.pool = nn.AvgPool1d(kernel_size=num_kernels, stride=num_kernels)
        self.hidden_layer = nn.Linear(num_kernels, num_hidden)
        self.output_layer = nn.Linear(num_hidden, num_classes)
        # self.output_layer = nn.Linear(num_kernels, num_classes)
        
        #TODO: define your model here

    def forward(self, x:torch.Tensor):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The network input of shape (batch_size, 1, in_channels, sequence_length)

        Returns
        ----------
        torch.Tensor
            The network output (softmax logits) of shape (batch_size, num_classes)
        """
        # TODO: implement the forward pass here
        x = torch.squeeze(x)
        x = self.conv_layer(x)
        # x = x.flatten(start_dim=1)
        # x = self.pool(x)
        x = F.relu(x)
        x = torch.mean(x, dim=2)
        
        #x = x.permute(0, 1) 
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        output = F.log_softmax(x, dim=-1)
        return output
        # pass


class KeyWordCNN2d(nn.Module):
    """
    """

    def __init__(self, num_classes, num_features, num_kernels, mem_depth, num_hidden=20):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes to predict
        num_features : int
            Number of signal features per time step
        num_kernels : int
            Number of convolution kernels
        mem_depth : int
            Memory depth = kernel size of the model
        """
        super().__init__()
        self.conv_layer = nn.Conv2d(1, num_kernels, (num_features, mem_depth))
        self.hidden_layer = nn.Linear(num_kernels, num_hidden)
        self.output_layer = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The network input of shape (batch_size, in_channels, num_features, sequence_length)

        Returns
        ----------
        torch.Tensor
            The network output (softmax logits) of shape (batch_size, num_classes)
        """
        # x = torch.squeeze(x)
        # print("forward")
        # print(x.shape)
        # x = torch.permute(x, (1,0,2,3))
        x = self.conv_layer(x)
        # print('huhu1', x.size())
        x = F.relu(x)
        # print('huhu2', x.size())
        x = torch.mean(x, dim=3).squeeze()
        # print('huhu3', x.size())
        x = self.hidden_layer(x)
        # print('huhu4', x.size())
        x = F.relu(x)
        # print('huhu5', x.size())
        x = self.output_layer(x)
        # print('huhu6', x.size())
        output = F.log_softmax(x, dim=-1)
        # print('huhu')
        return output
