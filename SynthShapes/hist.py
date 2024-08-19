__all__ = [
    'MatchHistogram'
]

# Standard imports
import torch
from torch import nn


class MatchHistogram(nn.Module):
    def __init__(self, mean=0.0, std=0.2, num_bins=256):
        """
        Histogram Matching Module to map the intensity values of an image to
        follow a normal distribution.

        Parameters
        ----------
        mean : float, optional
            Mean of the normal distribution, by default 0.0.
        std : float, optional
            Standard deviation of the normal distribution, by default 0.2.
        num_bins : int, optional
            Number of bins for histogram, by default 256.
        """
        super(MatchHistogram, self).__init__()
        self.mean = mean
        self.std = std
        self.num_bins = num_bins

    def calculate_cdf(self, hist):
        """Calculate the cumulative distribution function (CDF) for a
        histogram."""
        cdf = hist.cumsum(0)
        cdf_normalized = cdf / cdf[-1]
        return cdf_normalized

    def forward(self, source):
        """
        Forward pass to perform histogram matching.

        Parameters
        ----------
        source : torch.Tensor
            Source image (HxW), normalized between -1 and 1.

        Returns
        -------
        matched : torch.Tensor
            The transformed source image with histogram matching a normal
            distribution.
        """
        device = source.device

        # Normalize the source image to the range [0, 255] for histogram
        # computation
        source_normalized = ((source + 1) / 2 * 255).clamp(0, 255).long()

        # Compute the histogram and CDF of the source image
        src_hist = torch.histc(source_normalized.float(),
                               bins=self.num_bins, min=0, max=255).to(device)
        src_cdf = self.calculate_cdf(src_hist)

        # Create the normal distribution CDF
        normal_values = torch.linspace(-1, 1, self.num_bins, device=device)
        normal_cdf = torch.distributions.Normal(
            self.mean, self.std).cdf(normal_values)
        normal_cdf = normal_cdf / normal_cdf[-1]  # Normalize to range [0, 1]

        # Create a lookup table to map the pixel values
        lookup_table = torch.zeros(self.num_bins, device=device)
        for src_pixel in range(self.num_bins):
            normal_pixel = torch.searchsorted(normal_cdf, src_cdf[src_pixel])
            lookup_table[src_pixel] = normal_pixel

        # Apply the lookup table to the source image
        source_flat = source_normalized.flatten().long()
        matched_flat = lookup_table[source_flat]
        matched = matched_flat.view(source.shape).float()

        # Convert matched image back to the range [-1, 1]
        matched = matched / (self.num_bins - 1) * 2 - 1

        return matched
