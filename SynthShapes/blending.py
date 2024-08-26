__all__ = [
    'Blender'
]

# Standard imports
import torch
from torch.nn import Module


class Blender(Module):

    def __init__(self):
        """
        Blend tensors (shapes into image) using blending parameter.

        Parameters
        ----------
        image : tensor
            The intensity image tensor to blend shapes into.
        shapes : tensor
            The tensor with shapes to blend (each shape should have a unique
            ID).
        alpha : float
            Weight of the shape tensor. Larger magnitue
            --> more shape character
        """
        super(Blender, self).__init__()

    def forward(self, foreground: torch.Tensor, background: torch.Tensor,
                mask: torch.Tensor, alpha=0.3):
        """
        Perform blending operation.

        Parameters
        ----------
        image : tensor
            The intensity image tensor to blend shapes into.
        shapes : tensor
            The tensor with shapes to blend (each shape should have a unique
            ID).
        alpha : float
            Weight of the shape tensor. Larger magnitue
            --> more shape character
        """
        # Ensure image and shapes are both float
        foreground = foreground.float()
        background = background.float()

        # Init the blended tensor as a copy of the image tensor
        blended_tensor = torch.clone(background)
        blended_tensor[mask] = -(
            alpha * foreground[mask]) + (1 - alpha) * background[mask]

        return blended_tensor
