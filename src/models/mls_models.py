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

    def __init__(self, hidden_sizes: List[int]):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def fit(self, data, targets, batch_size=5, learning_rate=5e-3, max_epochs=300):
        raise NotImplementedError


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
        return np.array([sum(self.pbf(x_n, self.order) * self.alpha) for x_n in x])

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
        return np.array([sum(self.rbf(x_n, self.centers, self.lengthscale) * self.alpha[1:], self.alpha[0]) for x_n in x])

    def fit(self, data, targets):
        self.centers = np.linspace(data.min(), data.max(), self.order-1)
        return PolynomialModel.fit(self, data, targets)
    
    
    @staticmethod
    def rbf(x, c, w):
        return np.e ** -((x - c) ** 2 / w)
    
    
    def _make_feature_matrix(self, x, N):      
        phi_x = lambda x: np.append(np.array([1]), self.rbf(x, self.centers, self.lengthscale))
        
        A = np.array([phi_x(x_n) for x_n in x])
        return A

    def _mse(self, y_true, y_pred):
        return PolynomialModel._mse(y_true, y_pred)
    