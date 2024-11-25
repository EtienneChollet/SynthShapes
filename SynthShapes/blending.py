"""
## Overview

The SynthShapes.blending module offers simple tools for combining multiple
shapes, textures, or regions within an image to create seamless and realistic
synthetic data. This module is ideal for applications that require simple to
complex image compositions.
"""

__all__ = [
    'Blender'
]

# Standard imports
import cornucopia as cc
from cornucopia.random import make_range
import torch
from torch.nn import Module


class Blender(Module):
    """
    An [`nn.Module`][torch.nn.Module] for alpha blending ROIs of two tensors
    within a specified ROI mask, and according to the blending parameter
    `alpha`.

    !!! tip "Diagram"
        ```mermaid
            flowchart TB
                subgraph Inputs
                    foreground[Foreground Tensor]
                    background[Background Tensor]
                    mask[Mask: ROIs to blend]
                end
                foreground --standardize(μ=0, σ=1)--> standardized_foreground
                mask --> masked_shifting
                background --standardize(μ=0, σ=1)--> standardized_background
                standardized_foreground --> masked_shifting
                masked_shifting(Shift Foreground ROIs: add offset to masked
                elements in standardized foreground) --> shifted_rois
                shifted_rois["Shifted Foreground ROIs"] --> blender_function
                standardized_background --> blender_function
                blender_function("Blender Function: alpha blend ROIs") -->
                output[Output: Blended Tensor]

        ```
    """

    def __init__(
        self,
        alpha: float = cc.Uniform(0.5, 5),
        intensity_shift: float = 10
    ):
        """
        Parameters
        ----------
        alpha : Sampler or float
            Blending factor or sampler.
        intensity_shift: Sampler or float
            Intensity offset WRT mean=0, std=0 background.
        """
        super(Blender, self).__init__()
        self.alpha = cc.Uniform.make(make_range(0, alpha))
        self.intensity_shift = cc.Uniform.make(make_range(0, intensity_shift))

    def forward(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        mask: torch.Tensor
    ):
        """
        Forward pass of `Blender` to apply the blending operation and return
        the blended tensor.

        Parameters
        ----------
        foreground : tensor
            The tensor with shapes to blend (each shape should have a unique
            ID).
        background : tensor
            The intensity image (tensor) to blend shapes into.
        """
        # Sample params
        alpha = self.alpha()
        intensity_shift = self.intensity_shift()
        # Ensure image and shapes are both float
        foreground = foreground.float()
        background = background.float()
        # Background mean=0, std=1
        background -= background.mean()
        background /= background.std()
        # Foreground mean=0, std=1
        foreground[mask] -= foreground[mask].mean()
        foreground[mask] /= foreground[mask].std()
        foreground[mask] += intensity_shift
        # Perform blending operation
        foreground[mask] = (
            (background[mask] * (1 - alpha))
            + (foreground[mask] * alpha)
        )
        foreground[~mask] = background[~mask]
        return foreground
