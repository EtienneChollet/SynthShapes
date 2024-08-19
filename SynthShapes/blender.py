__all__ = [
    'Blender'
]

# Standard imports
from torch.nn import Module


class Blender(Module):

    def __init__(self):
        """
        Blend tensors (shapes into image) using blending parameter.
        """
        super(Blender, self).__init__()

    def forward(self, image, shapes, alpha=0.3):
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
        image = image.float()
        shapes = shapes.float()

        # Make a mask where shapes are non-zero
        mask = (shapes != 0).bool()

        # Init the blended tensor as a copy of the image tensor
        blended_tensor = image.clone()

        # Adjust the alpha scaling to ensure the result is less than p
        # We scale the difference between p and v instead of v directly
        blended_tensor[mask] = image[mask] - alpha * (
            image[mask] - shapes[mask])

        return blended_tensor
