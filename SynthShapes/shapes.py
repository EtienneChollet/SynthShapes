"""
## Overview

The `SynthShapes.shapes` module provides a comprehensive toolkit for generating
geometric shapes and patterns, tailored for synthetic image generation in
biomedical imaging and beyond. This module supports a variety of 2D and 3D
shapes with customizable parameters, enabling users to create realistic and
diverse datasets for machine learning and computer vision applications.

## Features

* **Shape Generators**: Includes predefined methods for creating common
geometric shapes like spheres, stars, toroids, rectangles, spheres, and cubes.

* **Customizability**: Allows fine-tuning of parameters such as size, position,
orientation, and intensity for each shape.

* **Multi-shape Compositions**: Supports combining multiple shapes into a
single image for more complex synthetic data.

* **Label Maps**: Since texturized data is synthesized from semantic label
maps, one-hot encoded labels are easily retreived for semantic segmentation
workflows.


"""
__all__ = [
    'MultiLobedBlobConfig',
    'MultiLobedBlobBase',
    'MultiLobedBlobSampler',
    'TorusBlobSampler'
]

# Standard Imports
import torch
from torch import nn
import cornucopia as cc
from cornucopia import Sampler
from cornucopia.random import make_range
from dataclasses import dataclass

# Custom Imports
from SynthShapes.texturizing import TexturizeLabels
from SynthShapes.blending import Blender


@dataclass
class MultiLobedBlobConfig:
    """
    Configuration container for MultiLobedBlobBase.

    Attributes
    ----------
    shape : list of int, optional
        Size of the 3D volume (assumed to be cubic). Default is [128, 128, 128].
    axis_length : Sampler, optional
        Range of the lengths for any given axis of the blob. Default is RandInt(3, 6).
    blob_density : Sampler, optional
        Density of blobs in the volume. Default is Fixed(0.05).
    n_blobs : Sampler, optional
        Maximum number of multi-lobed blobs. Default is RandInt(10, 25).
    n_lobes : Sampler, optional
        Sampler range for the number of lobes per blob. Default is RandInt(1, 5).
    sharpness : Sampler, optional
        Upper bound for factor controlling the squareness of the blobs.
        Note: +5 = mostly squares, 2 < sharpness < 4 = spheres, sharpness < 2 = stars.
        Default is Uniform(1, 3).
    jitter : Sampler, optional
        Maximum amount of jitter/raggedness to apply to the shape. Default is Uniform(0, 0.5).
    device : str, optional
        Device to perform computations on. Default is 'cuda'.
    """
    # shape: list = field(default_factory=lambda: [128, 128, 128])
    axis_length: Sampler = cc.RandInt.make(make_range(2, cc.RandInt(3, 6)))
    blob_density: Sampler = cc.Fixed(0.05)
    n_blobs: Sampler = cc.RandInt(10, 25)
    n_lobes: Sampler = cc.RandInt(1, 5)
    sharpness: Sampler = cc.Uniform(1, 3)
    jitter: Sampler = cc.Uniform(0, 0.5)
    return_mask: bool = True
    device: str = 'cuda'


@dataclass
class BlendingConfig:
    base_augmentation: nn.Sequential
    incoming_label_augmentation: nn.Sequential
    alpha: float = 0.5
    intensity_shift: float = 1


class MultiLobedBlobBase(nn.Module):
    """
    Base module for multi-lobed blob operations.

    Parameters
    ----------
    config : MultiLobedBlobConfig
        Configuration container for blob parameters.
    """
    def __init__(self, config: MultiLobedBlobConfig = MultiLobedBlobConfig):
        """
        Base module for multi-lobed blob operations.

        Parameters
        ----------
        config : MultiLobedBlobConfig
            Configuration container for blob parameters.
        """
        super(MultiLobedBlobBase, self).__init__()
        self.config = config
        self.current_label = 1

    def sample_axis_lengths(self):
        """
        Sample the axis lengths for all dimensions of a blob.

        Returns
        -------
        axis_lengths : torch.Tensor[int, int, int]
            Axis lengths of the blob (D, H, W).
        """
        axis_lengths = torch.tensor(
            [
                self.config.axis_length(),
                self.config.axis_length(),
                self.config.axis_length()
            ]
        )
        return axis_lengths

    def sample_centroid_coords(self, axis_lengths: torch.Tensor):
        """
        Sample the coordinates of the blob's centroid.

        Parameters
        ----------
        axis_lengths : torch.Tensor[int, int, int]
            Axis lengths of the blob (depth, height, width).

        Returns
        -------
        centroid_coords : tuple[int, int, int]
            Coordinates of the centroid (depth, height, width).
        """
        centroid_coords = (
            torch.randint(
                axis_lengths[0],
                self.shape[0] - axis_lengths[0],
                (1,), device=self.config.device).item(),
            torch.randint(
                axis_lengths[1],
                self.shape[1] - axis_lengths[1],
                (1,), device=self.config.device).item(),
            torch.randint(
                axis_lengths[2],
                self.shape[2] - axis_lengths[2],
                (1,), device=self.config.device).item()
            )
        return centroid_coords

    def check_overlap(self, axis_lengths: torch.Tensor, centroid_coords: tuple,
                      axes_list: list = [], centers: list = []):
        """
        Check if the newly sampled blob overlaps with existing blobs.

        This function computes the euclidean distance between the centroid of
        the new blobs and those of the existing blobs. The distance is
        compared to the sum of the largest axes of the blob, to ensure it is
        larger, ensuring no overlap exists between the blobs.

        Parameters
        ----------
        axis_lengths : torch.Tensor[int, int, int]
            Axis lengths of the new blob (depth, height, width).
        centroid_coords : tuple[int, int, int]
            Coordinates of the new blob's centroid (depth, height, width).
        axes_list : list
            A list of tuples representing the lengths of each axis for all
            blobs.
        centroid_coords : list
            A list of tuples representing the coordinates of the centroid of
            each existing blob.
        """
        overlap_exists = False
        # Iterate through all center coordinate and axis length pairs that have
        # been sampled
        for c, a in zip(centers, axes_list):
            # Calculate euclidean distance between the centroid of an existing
            # blob and this current blob.
            distance = torch.sqrt(
                torch.tensor((centroid_coords[0] - c[0]) ** 2
                             + (centroid_coords[1] - c[1]) ** 2
                             + (centroid_coords[2] - c[2]) ** 2,
                             device=self.config.device)
            )
            # Ensure the euclidean distance between centroids is larger than
            # the sum of the largest axes of both blobs, ensuring no overlap.
            if distance < max(a) + max(axis_lengths):
                # If the distance is less than sum of largest axes, overlap yes
                overlap_exists = True
                break
        return overlap_exists

    def sample_nonoverlapping_geometries(self):
        """
        Sample nonoverlapping bounding boxes by generating list of centers and
        axis lengths.

        This method creates two lists (center coordinates and axes lengths)
        that will be used to describe the bounding boxes for each blob to be
        sampled.

        Returns
        -------
        centers : list
            List of tuples representing the centroid coordinates for each
            bounding box (centroid_coords)
        axes_list : list
            List of tuples representing the length of each 3 axes for each
            geometry
        """
        # List of center coordinates for existing blobs
        centers = []
        # List of axes lengths for existing blobs
        axes_list = []

        # Iterate until we get self.n_blobs blobs.
        while len(centers) < self.n_blobs():
            # Sample a tuple of axis lengths.
            axis_lengths = self.sample_axis_lengths()
            # Sample a tuple of centroid coordinates.
            centroid_coords = self.sample_centroid_coords(axis_lengths)
            # Check if overlap exists between the newly sampled bounding box
            # and the existing ones.
            overlap_exists = self.check_overlap(axis_lengths, centroid_coords,
                                                axes_list, centers)
            if not overlap_exists:
                centers.append(centroid_coords)
                axes_list.append(axis_lengths.tolist())

        return centers, axes_list

    def _meshgrid_origin_at_centroid(self, center_coords: tuple[int]):
        """
        Create a distance map from centroid coordinates.

        This method creates a distance map/coordinate grid across the entirety
        of the 3D space where the origin of the coordinate grid is the center
        of the geometry (blob to be sampled).

        Parameters
        ----------
        center_coords : tuple[int]
            Tuple of centroid coordinates for a given blob to be sampled.
            (depth, height, width)

        Returns
        -------
        meshgrid : torch.Tensor
            Distance map/coordinate grid where the origin is the center of the
            blob

        """
        # create a coordinate grid for the entire volume, such that [0, 0, 0]
        # is the origin of the coordinate system, then shift the origin to the
        # coordinates of the centroid by subtracting them out.
        meshgrid = torch.meshgrid(
            torch.arange(
                self.shape[0], device=self.config.device
                ) - center_coords[0],
            torch.arange(
                self.shape[1], device=self.config.device
                ) - center_coords[1],
            torch.arange(
                self.shape[2], device=self.config.device
                ) - center_coords[2],
            indexing='ij')
        return meshgrid

    def _make_lobe_prob(self, meshgrid, axis_lengths):
        """
        Calculate the probability of belonging to a blob.

        This method calculates the probability of each point in space belonging
        to a particular blob by adding noise and other parameters to the
        distance map of a centroid across all 3D space. This method also
        considers the axis lengths. Sharpness controls how quickly the prob
        drops off as you move away from the centroid.

        Parameters
        ----------
        meshgrid : torch.Tensor
            Distance map/coordinate grid where the origin is the center of the
            blob
        axis_lengths : tuple
            Tuple of axis lengths for a single geometry.

        Returns
        -------
        lobe_prob : torch.Tensor
            3D tensor representing the probability map for the lobe.

        """
        # Handle the sharpness input
        # TODO: move uniform INIT to __init__()

        # Create noise for the entire distance map by sampling a uniform
        # distribution centered at 0 and whose stdev depends on self.jitter.
        # Prevents the blob from being perfectly symmetric.
        noise = torch.normal(0,
                             self.config.jitter(),
                             size=[
                                 self.shape[0],
                                 self.shape[1],
                                 self.shape[2]
                                 ],
                             device=self.config.device)
        # Calculates probability of a voxel belonging to a lobe at each point.
        # The absolute values of these ratios are raised to the power of
        # sharpness, which controls how quickly the probability drops off as
        # you move away from the centroid.
        lobe_prob = (torch.abs(meshgrid[0] / (axis_lengths[0] + noise)
                               ) ** self.config.sharpness() +
                     torch.abs(meshgrid[1] / (axis_lengths[1] + noise)
                               ) ** self.config.sharpness() +
                     torch.abs(meshgrid[2] / (axis_lengths[2] + noise)
                               ) ** self.config.sharpness())
        return lobe_prob

    def _make_lobe_from_prob(self, lobe_prob):
        """
        Create a label for an individual lobe.

        Parameters
        ----------
        lobe_prob : torch.Tensor
            3D tensor representing the probability map for the lobe.

        return : torch.Tensor
            Label mask with unique ID for lobe.
        """
        mask = (lobe_prob <= 1).to(torch.float32)
        mask = torch.masked_fill(mask, mask.bool(), self.current_label)
        self.current_label += 1
        return mask

    def make_shapes(self, shape):
        """
        Make shape labels.

        Returns
        -------
        imprint_tensor : torch.Tensor[int]
            Blobs with unique integer labels.
        """
        self.shape = shape
        self.max_blobs = self.config.blob_density() * (
            torch.prod(torch.tensor(self.shape).cuda(), 0)**0.5
            )
        self.max_blobs = torch.ceil(self.max_blobs).item()
        self.n_blobs = cc.RandInt(
            self.max_blobs//2, self.max_blobs
            )

        # self.depth, self.height, self.width = self.shape
        self.imprint_tensor = torch.zeros(
            self.shape,
            dtype=torch.float32,
            device=self.config.device
            )

        centers, axes_list = self.sample_nonoverlapping_geometries()
        for center, axes in zip(centers, axes_list):
            # Make tensor for blob (and all lobes)
            lobe_tensor = torch.zeros(
                shape, dtype=torch.float32, device=self.config.device)
            # Make lobes one by one
            for _ in range(self.config.n_lobes()):
                lobe_center_shift = torch.randint(
                    -axes[0]+1, axes[0]-1, (3,), device=self.config.device)
                shifted_center = (center[0] + lobe_center_shift[0],
                                  center[1] + lobe_center_shift[1],
                                  center[2] + lobe_center_shift[2])
                shifted_meshgrid = self._meshgrid_origin_at_centroid(
                    shifted_center)
                lobe_prob = self._make_lobe_prob(shifted_meshgrid, axes)
                lobe = self._make_lobe_from_prob(lobe_prob)
                lobe_tensor += lobe
            # Add all lobes to imprint tensor
            self.imprint_tensor[lobe_tensor > 0] = lobe_tensor[lobe_tensor > 0]

        return self.imprint_tensor.unsqueeze(0)


class MultiLobedBlobSampler(MultiLobedBlobBase):
    """
    PyTorch module to sample multi-lobed blob labels in a 3D tensor.

    Inherits from MultiLobedBlobBase.

    Parameters
    ----------
    config : MultiLobedBlobConfig
        Configuration container for blob parameters.
    """
    def __init__(self, config: MultiLobedBlobConfig = MultiLobedBlobConfig):
        """
        PyTorch module to sample multi-lobed blob labels in a 3D tensor.

        Inherits from MultiLobedBlobBase.

        Parameters
        ----------
        config : MultiLobedBlobConfig
            Configuration container for blob parameters.
        """
        super(MultiLobedBlobBase, self).__init__()
        self.config = config
        self.current_label = 1

    def forward(self, shape):
        """
        Apply blob-sampling operation.

        Returns
        -------
        shapes : torch.tensor[int]
            Blobs with unique integer labels.
        """
        blob_labels = self.make_shapes(shape)
        if self.config.return_mask:
            blob_mask = (blob_labels > 0).bool().to(self.config.device)
            return blob_labels, blob_mask
        else:
            return blob_labels


class MultiLobeBlobAugmentation(MultiLobedBlobBase):
    """
    PyTorch module to augment 3D data (C, D, H, W) by sampling and
    alpha-blending multi-lobed blobs.

    Inherits from MultiLobedBlobBase.

    Parameters
    ----------
    config : MultiLobedBlobConfig
        Configuration container for blob parameters.
    """
    def __init__(self,
                 config: MultiLobedBlobConfig = MultiLobedBlobConfig,
                 alpha: Sampler = cc.Uniform(0.25, 0.75),
                 intensity_shift: float = cc.Uniform(10, 20),
                 augmentation=None,
                 device='cuda'):
        """
        PyTorch module to augment 3D data (C, D, H, W) by sampling and
        alpha-blending multi-lobed blobs.

        MultiLobedBlobBase.

        Parameters
        ----------
        config : MultiLobedBlobConfig
            Configuration container for blob parameters.
        """
        super(MultiLobedBlobBase, self).__init__()
        self.config = config
        self.current_label = 1
        if augmentation is None:
            self.augmentation = cc.RandomGaussianMixtureTransform(mu=0)
        else:
            self.augmentation = augmentation

        self.alpha = cc.Uniform.make(make_range(0.5, alpha))
        self.intensity_shift = cc.Uniform.make(make_range(0, intensity_shift))
        self.blender = Blender(
            alpha=self.alpha,
            intensity_shift=self.intensity_shift
        )

    def forward(self, background_intensities: torch.Tensor):
        """
        Apply blob augmentation to input tensor.

        Parameters
        ----------
        incoming_tensor : torch.Tensor[float]
            Background tensor to augment of shape (C, D, H, W)

        Returns
        -------
        blended_blobs : torch.Tensor[float]
            Blobs alpha-blended into background.
        """
        shape = background_intensities.squeeze().shape
        incoming_shape_labels = self.make_shapes(shape)
        incoming_shape_mask = (incoming_shape_labels > 0).bool().to(
            self.config.device
            )
        if self.augmentation:
            incoming_shape_labels = self.augmentation(incoming_shape_labels)
            incoming_shape_labels[~incoming_shape_mask] = 0

        blended = self.blender(
            incoming_shape_labels.to(self.config.device),
            background_intensities.to(self.config.device),
            incoming_shape_mask.to(self.config.device)
        )

        if self.config.return_mask:
            return blended, incoming_shape_mask
        else:
            return blended

# Example useage
# x = torch.ones((1, 1, 128, 128, 128))
# augmented = MultiLobeBlobAugmentation()(x)


class TorusBlobBase(nn.Module):
    """
    Base PyTorch module for torus blob operations.

    Parameters
    ----------
    major_radius_range : list of int, optional
        Range of lengths for the major radius of the torus. Default is [10, 20]
    minor_radius_range : list of int, optional
        Range of lengths for the minor radius of the torus. Default is [3, 6].
    n_blobs : int, optional
        Maximum number of tori to generate. Default is 10.
    jitter : float, optional
        Maximum amount of jitter to apply to the shape. Default is 0.5.
    device : str, optional
        Device to run the tensor operations ('cuda' or 'cpu'). Default is
        'cuda'.
    shape : int, optional
        Size of the 3D volume (assumed to be cubic). Default is 128.
    """
    def __init__(self, major_radius_range=[10, 20], minor_radius_range=[3, 6],
                 n_blobs: Sampler = cc.RandInt(10, 25),
                 jitter: Sampler = cc.Uniform(0, 0.5),
                 device='cuda', shape=128):
        """
        Base PyTorch module for torus blob operations.

        Parameters
        ----------
        major_radius_range : list of int, optional
            Range of lengths for the major radius of the torus. Default is
            [10, 20]
        minor_radius_range : list of int, optional
            Range of lengths for the minor radius of the torus. Default is
            [3, 6].
        n_blobs : int, optional
            Maximum number of tori to generate. Default is 10.
        jitter : float, optional
            Maximum amount of jitter to apply to the shape. Default is 0.5.
        device : str, optional
            Device to run the tensor operations ('cuda' or 'cpu'). Default is
            'cuda'.
        shape : int, optional
            Size of the 3D volume (assumed to be cubic). Default is 128.
        """
        super(TorusBlobBase, self).__init__()
        self.major_radius_range = major_radius_range
        self.minor_radius_range = minor_radius_range
        self.jitter = cc.Uniform.make(make_range(0, jitter))
        self.device = device
        self.shape = [shape, shape, shape]
        self.depth, self.height, self.width = self.shape
        self.n_blobs = cc.RandInt.make(make_range(1, n_blobs))
        self.imprint_tensor = torch.zeros(self.shape, dtype=torch.float32,
                                          device=self.device)
        self.current_label = 1

    def sample_radii(self):
        """
        Sample the major and minor axes radii.

        Returns
        -------
        major_radius : float
            The magnitude of the radius of the larger axis.
        minor_radius
            The magnitude of the radius of the smaller axis.
        """
        major_radius = torch.randint(
            self.major_radius_range[0],
            self.major_radius_range[1] + 1,
            (1,),
            device=self.device).item()
        minor_radius = torch.randint(
            self.minor_radius_range[0],
            self.minor_radius_range[1] + 1,
            (1,),
            device=self.device).item()
        return major_radius, minor_radius

    def sample_centroid_coords(self, major_radius):
        """
        Sample the coordinates of the centroid.

        Returns
        -------
        centroid_coords : tuple
            Coordinates of the centroid (depth, height, width).
        """
        centroid_coords = (
            torch.randint(
                major_radius,
                self.depth - major_radius,
                (1,), device=self.device).item(),
            torch.randint(
                major_radius,
                self.height - major_radius,
                (1,), device=self.device).item(),
            torch.randint(
                major_radius,
                self.width - major_radius,
                (1,), device=self.device).item()
            )
        return centroid_coords

    def check_overlap(self, major_radius: float, centroid_coords: tuple,
                      radii_list: list = [], centers: list = []):
        """
        Check if a newly sampled blob overlaps with existing blobs.

        This function computes the euclidean distance between the centroid of
        a newly sampled blob, and the centroids of previously sampled blobs. It
        then checks if the distance is less than the sum of the radii,
        indicating an overlap between blobs.

        Parameters
        ----------
        major_radius : float
            The radius of the largest dimension of the newly sampled blob.
        centroid_coords : tuple
            Coordinates of the centroid (depth, height, width).
        radii_list : list[float]
            A list of tuples of the radii belonging to blobs that have already
            been sampled.
        centers : list[float]
            A list of tuple of the centers belonging to blobs that have already
            been sampled.

        Returns
        -------
        overlap_exists : bool
            Whether overlap exists.
        """
        overlap_exists = False
        # Iterate through the centers and radii of all blobs that have been
        # sampled
        for c, r in zip(centers, radii_list):
            # Calculate euclidean distance between the centroids of the
            # current and existing blobs
            distance = torch.sqrt(
                torch.tensor(
                    (centroid_coords[0] - c[0]) ** 2
                    + (centroid_coords[1] - c[1]) ** 2
                    + (centroid_coords[2] - c[2]) ** 2,
                    device=self.device)
                    )
            # Compare euclidean distance between centroids to the sum of the
            # maximum radius of the existing blob and the max radius this blob.
            if distance < r[0] + major_radius:
                # If the distance is less than the sum of radii, overlap exists
                overlap_exists = True
                break
        return overlap_exists

    def sample_nonoverlapping_geometries(self):
        centers = []
        radii_list = []

        # Sample a blob if we don't have self.n_blobs blobs
        while len(centers) < self.n_blobs():
            # Get major and minor axes
            major_radius, minor_radius = self.sample_radii()
            # Sample coords ensuring max dimension of shape resides in bbox
            centroid_coords = self.sample_centroid_coords(major_radius)
            # Check if overlap exists between this blob and the bounds of
            # existing blobs
            overlap_exists = self.check_overlap(major_radius, centroid_coords,
                                                radii_list, centers)
            if not overlap_exists:
                centers.append(centroid_coords)
                radii_list.append((major_radius, minor_radius))

        return centers, radii_list

    def _meshgrid_origin_at_centroid(self, center_coords):
        meshgrid = torch.meshgrid(
            torch.arange(
                self.depth, device=self.device).float() - center_coords[0],
            torch.arange(
                self.height, device=self.device).float() - center_coords[1],
            torch.arange(
                self.width, device=self.device).float() - center_coords[2],
            indexing='ij')
        return meshgrid

    def _rotation_matrix(self, angles):
        """Create a rotation matrix for a given set of angles (in radians)."""
        alpha, beta, gamma = angles
        R_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(alpha), -torch.sin(alpha)],
                            [0, torch.sin(alpha), torch.cos(alpha)]],
                           device=self.device).float()

        R_y = torch.tensor([[torch.cos(beta), 0, torch.sin(beta)],
                            [0, 1, 0],
                            [-torch.sin(beta), 0, torch.cos(beta)]],
                           device=self.device).float()

        R_z = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                            [torch.sin(gamma), torch.cos(gamma), 0],
                            [0, 0, 1]], device=self.device).float()

        R = torch.matmul(R_z, torch.matmul(R_y, R_x))
        return R

    def _apply_rotation(self, meshgrid, rotation_matrix):
        """Apply the rotation matrix to the meshgrid."""
        grid_shape = meshgrid[0].shape
        grid_flat = torch.stack(meshgrid, dim=-1).reshape(-1, 3)
        rotated_grid_flat = torch.matmul(grid_flat, rotation_matrix)
        rotated_grid = rotated_grid_flat.reshape(grid_shape + (3,))
        return [rotated_grid[..., i] for i in range(3)]

    def _make_torus_prob(self, meshgrid, major_radius, minor_radius):
        noise = torch.normal(0, self.jitter(),
                             size=[self.depth, self.height, self.width],
                             device=self.device)
        radial_distance = torch.sqrt(
            (meshgrid[0] + noise) ** 2 + (meshgrid[1] + noise) ** 2
        ) - major_radius
        torus_prob = (
            radial_distance ** 2 + (
                meshgrid[2] + noise) ** 2) / minor_radius ** 2
        return torus_prob

    def _make_torus_from_prob(self, torus_prob):
        mask = (torus_prob <= 1).to(torch.float32)
        mask = torch.masked_fill(mask, mask.bool(), self.current_label)
        self.current_label += 1
        return mask

    def make_shapes(self):
        centers, radii_list = self.sample_nonoverlapping_geometries()
        for center, (major_radius, minor_radius) in zip(centers, radii_list):
            meshgrid = self._meshgrid_origin_at_centroid(center)

            # Randomly sample rotation angles
            # Random angles between 0 and 2*pi
            angles = torch.rand(3) * 2 * torch.pi
            rotation_matrix = self._rotation_matrix(angles).float()

            # Apply rotation to meshgrid
            rotated_meshgrid = self._apply_rotation(meshgrid, rotation_matrix)

            torus_prob = self._make_torus_prob(
                rotated_meshgrid, major_radius, minor_radius)
            torus = self._make_torus_from_prob(torus_prob)
            self.imprint_tensor[torus > 0] = torus[torus > 0]

        return self.imprint_tensor.unsqueeze(0)


class TorusBlobSampler(TorusBlobBase):
    """
    PyTorch module to sample torus blob labels in a 3D tensor.

    Inherits from TorusBlobBase.

    Parameters
    ----------
    major_radius_range : list of int, optional
        Range of lengths for the major radius of the torus. Default is
        [10, 20].
    minor_radius_range : list of int, optional
        Range of lengths for the minor radius of the torus. Default is [3, 6].
    n_blobs : int, optional
        Maximum number of tori to generate. Default is 10.
    jitter : float, optional
        Maximum amount of jitter to apply to the shape. Default is 0.5.
    device : str, optional
        Device to run the tensor operations ('cuda' or 'cpu'). Default is
        'cuda'.
    shape : int, optional
        Size of the 3D volume (assumed to be cubic). Default is 128.
    """
    def __init__(self, major_radius_range=[10, 20], minor_radius_range=[3, 6],
                 n_blobs: Sampler = cc.RandInt(10, 25),
                 jitter: Sampler = cc.Uniform(0, 0.5),
                 device: str = 'cuda', shape: int = 128,
                 return_mask: bool = True
                 ):
        """
        PyTorch module to sample torus blob labels in a 3D tensor.

        Inherits from TorusBlobBase.

        Parameters
        ----------
        major_radius_range : list of int, optional
            Range of lengths for the major radius of the torus. Default is
            [10, 20].
        minor_radius_range : list of int, optional
            Range of lengths for the minor radius of the torus. Default is
            [3, 6].
        n_blobs : int, optional
            Maximum number of tori to generate. Default is 10.
        jitter : float, optional
            Maximum amount of jitter to apply to the shape. Default is 0.5.
        device : str, optional
            Device to run the tensor operations ('cuda' or 'cpu'). Default is
            'cuda'.
        shape : int, optional
            Size of the 3D volume (assumed to be cubic). Default is 128.
        """
        super(TorusBlobBase, self).__init__()
        self.major_radius_range = major_radius_range
        self.minor_radius_range = minor_radius_range
        self.jitter = cc.Uniform.make(make_range(0, jitter))
        self.device = device
        self.shape = [shape, shape, shape]
        self.return_mask = return_mask
        self.depth, self.height, self.width = self.shape
        self.n_blobs = cc.RandInt.make(make_range(1, n_blobs))
        self.imprint_tensor = torch.zeros(self.shape, dtype=torch.float32,
                                          device=self.device)
        self.current_label = 1

    def forward(self):
        """
        Apply torus sampling operation.

        Returns
        -------
        shapes : torch.Tensor[int]
            Tori with unique integer labels.
        """
        labels = self.make_shapes()
        if self.return_mask:
            mask = (labels > 0).bool().to(self.device)
            return labels, mask
        else:
            return labels


class TorusBlobAugmentation(TorusBlobBase):
    """
    A module to augment 3D data (B, C, D, H, W) by sampling and
    alpha-blending tori.

    Inherits from TorusBlobBase.

    Parameters
    ----------
    major_radius_range : list of int, optional
        Range of lengths for the major radius of the torus. Default is
        [10, 20].
    minor_radius_range : list of int, optional
        Range of lengths for the minor radius of the torus. Default is [3, 6].
    n_blobs : int, optional
        Maximum number of tori to generate. Default is 10.
    jitter : float, optional
        Maximum amount of jitter to apply to the shape. Default is 0.5.
    device : str, optional
        Device to run the tensor operations ('cuda' or 'cpu'). Default is
        'cuda'.
    shape : int, optional
        Size of the 3D volume (assumed to be cubic). Default is 128.
    """
    def __init__(self,
                 shape=128,
                 major_radius_range=[10, 20],
                 minor_radius_range=[3, 6],
                 n_blobs=cc.RandInt(10, 25),
                 jitter: Sampler = cc.Uniform(0, 0.5),
                 return_mask: bool = False,
                 alpha: Sampler = cc.Uniform(0.25, 0.75),
                 intensity_shift: float = cc.Uniform(10, 20),
                 augmentation=None,
                 device='cuda',
                 ):
        super(TorusBlobBase, self).__init__()
        """
        A module to augment 3D data (B, C, D, H, W) by sampling and
        alpha-blending tori.

        Parameters
        ----------
        major_radius_range : list of int, optional
            Range of lengths for the major radius of the torus. Default is
            [10, 20].
        minor_radius_range : list of int, optional
            Range of lengths for the minor radius of the torus. Default is
            [3, 6].
        n_blobs : int, optional
            Maximum number of tori to generate. Default is 10.
        jitter : float, optional
            Maximum amount of jitter to apply to the shape. Default is 0.5.
        device : str, optional
            Device to run the tensor operations ('cuda' or 'cpu'). Default is
            'cuda'.
        shape : int, optional
            Size of the 3D volume (assumed to be cubic). Default is 128.
        alpha : list[float, float]
            Range of alpha magnitude for alpha blending. Default is
            [0.25, 0.75]
        intensity
        return_mask : bool
            Optionally return torus mask. Default is False
        """
        self.shape = [shape, shape, shape]
        self.major_radius_range = major_radius_range
        self.minor_radius_range = minor_radius_range
        self.n_blobs = cc.RandInt.make(make_range(1, n_blobs))
        self.jitter = cc.Uniform.make(make_range(0, jitter))
        self.return_mask = return_mask
        self.device = device
        self.depth, self.height, self.width = self.shape
        self.imprint_tensor = torch.zeros(
            self.shape, dtype=torch.float32, device=self.device
            )
        self.current_label = 1
        if augmentation is None:
            self.augmentation = cc.RandomGaussianMixtureTransform(mu=0)
        else:
            self.augmentation = augmentation

        self.alpha = cc.Uniform.make(make_range(0.5, alpha))
        self.intensity_shift = cc.Uniform.make(make_range(0, intensity_shift))
        self.blender = Blender(
            alpha=self.alpha,
            intensity_shift=self.intensity_shift
        )

    def forward(self, background_intensities: torch.Tensor):
        """
        Apply torus augmentation to input tensor.

        Parameters
        ----------
        background_intensities : torch.Tensor[float]
            Background tensor to augment of shape (C, D, H, W)

        Returns
        -------
        blended_tori : torch.Tensor[float]
            Tori alpha-blended into background.
        """

        incoming_shape_labels = self.make_shapes()
        incoming_shape_mask = (incoming_shape_labels > 0).bool().to(
            self.device
            )
        if self.augmentation:
            incoming_shape_labels = self.augmentation(incoming_shape_labels)
            incoming_shape_labels[~incoming_shape_mask] = 0

        blended = self.blender(
            incoming_shape_labels.to(self.device),
            background_intensities.to(self.device),
            incoming_shape_mask.to(self.device)
        )

        if self.return_mask:
            return blended, incoming_shape_mask
        else:
            return blended

# Example useage
# x = torch.ones((1, 1, 128, 128, 128))
# augmented = TorusBlobAugmentation()(x)

# TODO: Make torodial knot sampler
# TODO: Make cylendar sampler
# TODO: Make cone sampler
