__all__ = [
    'MinMaxScaler'
]

import torch
import torch.nn as nn


class MinMaxScaler(nn.Module):
    def __init__(self, lower_bound: float = 0.01, upper_bound: float = 0.99):
        """
        A torch.nn.Module subclass that scales an input tensor to a specified
        range.

        Parameters
        ----------
        lower_bound : float
            The lower bound of the scaling interval. Default is 0.1.
        upper_bound : float
            The upper bound of the scaling interval. Default is 0.9.
        """
        super(MinMaxScaler, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to scale the input tensor X to the interval
        [lower_bound, upper_bound].

        Parameters
        ----------
        X : torch.Tensor
            Input tensor with arbitrary floating point values.

        Returns
        -------
        torch.Tensor
            Scaled tensor with values in the interval
            [lower_bound, upper_bound].
        """
        X_min = X.min()
        X_max = X.max()

        # Prevent division by zero if X_min == X_max
        if X_min == X_max:
            return torch.full_like(X, self.lower_bound)

        # Scale the tensor to the range [lower_bound, upper_bound]
        X_scaled = self.lower_bound + (X - X_min) * (
            self.upper_bound - self.lower_bound) / (X_max - X_min)

        return X_scaled
