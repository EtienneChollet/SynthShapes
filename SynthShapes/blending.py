__all__ = [
    'Blender'
]

# Standard imports
import cornucopia as cc
from cornucopia.random import make_range
import torch
from torch.nn import Module


class Blender(Module):

    def __init__(
        self,
        alpha: float = cc.Uniform(0.5, 5),
        intensity_shift: float = 10
    ):
        """
        Blend tensors (shapes into image) using blending parameter.

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
        Perform blending operation.

        Parameters
        ----------
        foreground : tensor
            The tensor with shapes to blend (each shape should have a unique
            ID).
        background : tensor
            The intensity image (tensor) to blend shapes into.
        alpha : float
            Weight of the shape tensor. Larger magnitude = more blob character.
            Tip: alpha > 1 maintains blob texture well.
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
