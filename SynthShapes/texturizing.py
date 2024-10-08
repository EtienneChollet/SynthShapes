__all__ = [
    'LabelsToIntensities',
    'ParenchymaSynthesizer'
]

import torch
import cornucopia as cc
from torch.nn import Module
from cornucopia.random import make_range
from SynthShapes.blending import Blender
from cornucopia.labels import RandomGaussianMixtureTransform

from .utils import MinMaxScaler


class LabelsToIntensities(Module):

    def __init__(self, mu=1, sigma=2, min=0.1, max=0.5, transform=None):
        """
        Convert a set of labels with unique IDs into intensities.

        Parameters
        ----------
        mu : float
            Mean of GMM.
        sigma : float
            Sigma of GMM.
        min : float
            Minimum value of the output tensor (except background zeros)
        max : float
            Maximum value of output tensor.
        transform : torch.nn.Module
            Single transform or moduledict
        """
        super(LabelsToIntensities, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max
        if transform is None:
            self.transform = RandomGaussianMixtureTransform(mu=1, sigma=2)
        else:
            self.transform = transform

    def forward(self, labels):
        """
        Apply the transformation

        Parameters
        ----------
        labels : torch.Tensor
            Labels with unique int ID's. Shape: (x, y, z)
        """

        # Create a mask for all labels (background = 0, labels = 1)
        label_mask = (torch.clone(labels) != 0)
        # Assign intensities by applying transform
        intensities = self.transform(labels)
        # Invert mask and zero all background values
        intensities[~label_mask] = 0
        

        return intensities


class TexturizeLabels(Module):

    def __init__(self, sigma: float = 2, intensity: float = 1,
                 transform=None):
        """
        Parameters
        ----------
        sigma: Sampler or float
            Standard deviation for label textures.
        intensity 
        """
        super(TexturizeLabels, self).__init__()
        self.sigma = cc.Uniform.make(make_range(0, sigma))
        self.intensity = cc.Uniform.make(make_range(0, intensity))
        if transform is None:
            self.transform = torch.nn.Sequential(
                cc.RandomGaussianMixtureTransform(sigma=self.sigma),
                cc.RandomGammaNoiseTransform(),
                cc.MulFieldTransform(vmin=0.1, vmax=0.75)
                )
        else:
            self.transform = transform

    def forward(self, label_map):
        texturized = self._apply_transform(label_map.to(torch.int))
        texturized = self._reassign_intensities(label_map, texturized)
        return texturized

    def _apply_transform(self, label_map: torch.Tensor):
        texturized = self.transform(label_map)
        return texturized

    def _reassign_intensities(self, label_map: torch.Tensor,
                              texturized: torch.Tensor):
        """
        Parameters
        ----------
        label_map : torch.Tensor
            label map of unique integer labels.
        """
        background_label_ids = torch.unique(label_map)
        texturized_adjusted = torch.clone(texturized)
        # TODO: Avoid zeros
        for label in background_label_ids:
            if label > 0:
                mask = (label_map == label).bool()
                average_intensity = self.intensity()  # cc.Uniform(0, 1)()
                bls = texturized[mask] - average_intensity
                texturized_adjusted[mask] -= bls.mean()
        return texturized_adjusted


class ParenchymaSynthesizer(Module):
    def __init__(self):
        """
        A torch.nn.Module subclass that synthesizes a background tensor
        by applying a series of transformations.
        """
        super(ParenchymaSynthesizer, self).__init__()

        # Define the transformations to be applied sequentially
        self.parenchyma_transform = torch.nn.Sequential(
            cc.RandomGaussianMixtureTransform(mu=1, sigma=2),
            MinMaxScaler(),
            cc.RandomGammaNoiseTransform(),
            MinMaxScaler(),
            cc.MulFieldTransform(vmin=0.1, vmax=0.75),
            MinMaxScaler(),
        )

        # Define the final quantile transform
        self.quantile_transform = cc.QuantileTransform()

        # Define the blender
        self.blender = Blender()

        # Define the initial smooth label map generator
        self.random_smooth_label_map = cc.RandomSmoothLabelMap()

        # Final MinMaxScaler to normalize the background
        self.final_scaler = MinMaxScaler()

    def forward(self, intensities_list: list, alpha: float = 0.4
                ) -> torch.Tensor:
        """
        Synthesizes the background by applying the defined transformations
        and blending multiple intensity tensors.

        Parameters
        ----------
        intensities_list : list of torch.Tensor
            A list of tensors with intensities to blend into the background.
        alpha : float
            The blending parameter controlling the influence of intensities in
            the background.

        Returns
        -------
        torch.Tensor
            The synthesized background tensor.
        """
        # Initialize the parenchyma tensor with ones and apply the smooth
        # label map.
        # Assuming all tensors are on same device.
        device = intensities_list[0].device
        parenchyma = torch.ones_like(intensities_list[0]).to(device)
        parenchyma = self.random_smooth_label_map(parenchyma)
        parenchyma += 1

        # Apply the sequential transformations
        parenchyma = self.parenchyma_transform(parenchyma)

        # Blend each intensity tensor in the list with the parenchyma
        for intensities in intensities_list:
            parenchyma = self.blender(
                foreground=intensities,
                background=parenchyma,
                mask=(intensities > 0).bool(),
                alpha=alpha)

        # Apply final scaling and quantile transform
        parenchyma = self.final_scaler(parenchyma)
        parenchyma = self.quantile_transform(parenchyma)

        return parenchyma
