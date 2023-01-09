from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset


class MLPModel(nn.Module):
    """
    A simple feed-forward network with one input/output unit, two hidden layers and tanh activation.

    Attributes
    ----------
    hidden_sizes : int
        Number of hidden units per hidden layer
    """

    def __init__(
            self, hidden_sizes: List[int],
            activation_function=torch.tanh):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.input_layer = nn.Linear(1, hidden_sizes[0])
        self.hidden_layers = [
            nn.Linear(in_dim, out_dim) for in_dim,
            out_dim in zip(hidden_sizes[: -1],
                           hidden_sizes[1:])]
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.activation_function = activation_function

    def __str__(self):
        return "MLPModel"

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        # get input into shape (batch_size, 1)
        x = x.view((-1, 1))

        x = self.activation_function(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))

        x = self.output_layer(x)
        return x

    def predict(self, x):
        output = self(x)
        return output.detach().numpy()

    def fit(
            self, data, targets, batch_size=5, learning_rate=5e-3,
            max_epochs=300):
        """Train the network on the given data.

        Parameters
        ----------
        data : numpy.array
            The input training samples
        targets : numpy.array
            The output training samples
        batch_size : int
            Batch size for mini-batch gradient descent training
        learning_rate : float
            The learning rate for the optimizer
        max_epochs : int
            Maximum number of training epochs

        Returns
        ----------
        list
            List of training set MSE scores for each training epoch
        """
        # define the loss function (MSE) and the optimizer (Adam)
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # create a data loader that takes care of batching our training set
        data = torch.FloatTensor(data[:, np.newaxis]) if len(
            data.shape) < 2 else torch.FloatTensor(data)
        targets = torch.FloatTensor(targets[:, np.newaxis]) if len(
            targets.shape) < 2 else torch.FloatTensor(targets)
        training_set = TensorDataset(data, targets)
        train_loader = DataLoader(
            training_set, batch_size=batch_size, shuffle=True)

        # train the network
        train_loss_list = []
        for i in range(max_epochs):
            for batch in train_loader:
                batch_data = batch[0]
                batch_targets = batch[1]

                # compute batch loss
                predictions = self(batch_data)
                batch_loss = loss(predictions, batch_targets)

                # optimize the network parameters
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # record the training set MSE after each epoch; `with torch.no_grad()` makes sure that we do not record
            # gradient information of anything inside this environment
            with torch.no_grad():
                predictions = self(training_set[:][0])
                train_loss = loss(predictions, training_set[:][1])
                train_loss_list.append(train_loss.detach().numpy())

            # log the training loss
            # print(f'Epoch {i} training loss is {train_loss:.2f}', end='\r')
            # if i % 50 == 0:
            #     print('')

        return train_loss_list


class PolynomialModel:
    def __init__(self, order: int):
        self.order: int = order
        self.alpha: np.ndarray

    def __str__(self):
        return "PolynomialModel"

    def predict(self, x):
        """Compute the network output for a given input and return the result as numpy array.

        Parameters
        ----------
        x : float or numpy.array or torch.Tensor
            The model input

        Returns
        ----------
        numpy.array
            The model output
        """
        return np.array([sum(self.pbf(x_n, self.order) * self.alpha)
                         for x_n in x])

    def fit(self, data, targets):
        """Fit coefficients of the polynomial model on the given data.

        Parameters
        ----------
        data : numpy.array
            The input training samples
        targets : numpy.array
            The output training samples
        Returns
        ----------
        list
            The training set MSE score
        """
        A = self._make_feature_matrix(data, self.order)
        alpha = np.linalg.pinv(A) @ targets.reshape(-1, 1)
        self.alpha = alpha.ravel()

        return self._mse(targets, self.predict(data))

    def pbf(self, x, k):
        return np.tile(x, (k + 1,)) ** np.arange(k+1)

    def _make_feature_matrix(self, x, N):
        A = np.array([self.pbf(x_n, N) for x_n in x])
        return A

    @staticmethod
    def _mse(y_true, y_pred):
        N = len(y_true)
        assert N == len(y_pred)

        return np.mean((y_true - y_pred) ** 2)


class RBFModel:
    def __init__(self, order: int, lengthscale: float = 10.):
        self.order = order
        self.lengthscale = lengthscale
        self.alpha: np.ndarray
        self.centers: np.ndarray

    def __str__(self):
        return "RBFModel"

    def predict(self, x):
        return np.array([
            sum(
                self.rbf(x_n, self.centers, self.lengthscale) *
                self.alpha[1:],
                self.alpha[0]) for x_n in x])

    def fit(self, data, targets):
        self.centers = np.linspace(data.min(), data.max(), self.order-1)
        return PolynomialModel.fit(self, data, targets)

    @staticmethod
    def rbf(x, c, w):
        return np.e ** -((x - c) ** 2 / w)

    def _make_feature_matrix(self, x, N):
        def phi_x(x): return np.append(
            np.array([1]), self.rbf(x, self.centers, self.lengthscale))

        A = np.array([phi_x(x_n) for x_n in x])
        return A

    def _mse(self, y_true, y_pred):
        return PolynomialModel._mse(y_true, y_pred)
