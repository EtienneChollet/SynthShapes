# Standard imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Custom imports
from .utils import ensure_5d_tensor

class MinimumFilter3D(nn.Module):
    """
    A module to apply a minimum filter to 3D volumes using 3x3x3 patches.

    Parameters
    ----------
    kernel_size : int
        Size of the neighborhood from which the minimum is computed. Must be
        an odd number, typically 3 for a 3x3x3 filter.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, volume):
        """
        Apply the minimum filter to an input 3D volume.

        Parameters
        ----------
        volume : torch.Tensor
            Input 4D tensor representing the volume. Should be of shape
            [batch_size, channels, depth, height, width].

        Returns
        -------
        torch.Tensor
            Output 4D tensor after applying the minimum filter. Shape matches
            the input shape.
        """
        # Reshaping tensor as needed
        volume = ensure_5d_tensor(volume)

        # Apply a max pooling with a negative sign to simulate a minimum filter
        volume_neg = -volume
        min_filtered = -F.max_pool3d(
            volume_neg, kernel_size=self.kernel_size, padding=self.padding,
            stride=1)

        return min_filtered


class GaussianSmoothing3D(nn.Module):
    """
    A module to apply Gaussian smoothing to 3D volumes using 3D convolution.

    This class extends `nn.Module` to handle 3D data, applying a Gaussian
    smoothing filter across the input tensor using a 3D kernel.

    Parameters
    ----------
    kernel_size : int
        Size of the Gaussian kernel. Must be an odd number, typically 3 or 5.
    sigma : float
        Standard deviation of the Gaussian kernel.
    """
    def __init__(self, kernel_size=3, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

        # Create the 3D Gaussian kernel
        self.kernel = self.create_gaussian_kernel(kernel_size, sigma)

    def forward(self, volume):
        """
        Apply the Gaussian smoothing filter to an input 3D volume.

        Parameters
        ----------
        volume : torch.Tensor
            Input 4D tensor representing the volume. Should be of shape
            [batch_size, channels, depth, height, width].

        Returns
        -------
        torch.Tensor
            Output 4D tensor after applying Gaussian smoothing. Shape matches 
            the input shape.
        """
        # Reshaping tensor as needed
        volume = ensure_5d_tensor(volume)

        # Apply Gaussian filter using 3D convolution
        padding = (self.padding, self.padding, self.padding, self.padding,
                   self.padding, self.padding)
        padded_volume = F.pad(volume, padding, mode='reflect')
        volume = F.conv3d(padded_volume, self.kernel, padding=0,
                          groups=volume.shape[1])
        return volume

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        Create the 3D Gaussian kernel.

        Parameters
        ----------
        kernel_size : int
            Size of Gaussian kernel.
        sigma : float
            Standard deviation of the Gaussian kernel.

        Returns
        -------
        torch.Tensor
            5D tensor representing the 3D Gaussian kernel for use in conv3d.
        """
        # Create a coordinate grid centered at zero
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        grid = torch.stack(torch.meshgrid(coords, coords, coords), -1)

        # Calculate the Gaussian function
        kernel = torch.exp(-((grid ** 2).sum(-1) / (2 * sigma ** 2))).cuda()

        # Normalize the kernel so that the sum of all elements is 1
        kernel = kernel / kernel.sum()

        # Reshape to 5D tensor for conv3d
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        kernel = kernel.repeat(1, 1, 1, 1, 1)

        return kernel
