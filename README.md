![SynthShapes](https://github.com/user-attachments/assets/2e6a3aa6-3e87-4762-aa8d-28c18f8ca6df)



# SynthShapes
SynthShapes is a PyTorch toolbox for generating synthetic 3D shapes, tailored specifically for augmenting biomedical imaging datasets. It allows users to create complex, realistic shapes for use in machine learning pipelines, especially when training data is scarce or needs augmentation.

# Features
* Generate a variety of 3D shapes (e.g., blobs, toroids).
* Easily customizable parameters (size, shape, intensity).
* Ready-to-go augmentation modules for biomedical image datasets (real or synth).


# Installation
It is suggested that you create and activate a new mamba environment with python 3.9. You can learn how to install mamba by following the instructions provided in the [Miniforge repo](https://github.com/conda-forge/miniforge).

```bash
mamba create -n synthshapes python=3.9
mamba activate synthshapes
```

Now let's install install SynthShapes from pypi! It's as easy as that!

```bash
pip install SynthShapes
```

# Usage

```python
import torch
import cornucopia as cc
import matplotlib.pyplot as plt
from SynthShapes.shapes import MultiLobeBlobAugmentation

# Generate the background (or medical imaging volume)
volumetric_data = torch.randn((1, 64, 64, 64))

# Augment the data and return the blob mask (for, perhaps, supervised segmentation)
augmented_volumetric_data, blob_mask = MultiLobeBlobAugmentation(
    alpha=cc.Uniform(0.5, 0.75),
    intensity_shift=cc.Uniform(2, 20),
    return_mask=True
    )(volumetric_data)

# Visualize one slice
plt.imshow(augmented_volumetric_data[0, 32], cmap='gray')
plt.show()
```

## Available Shapes
Although there are currently only two available shapes to choose from, they span a wide range of configurations, thanks to their jitter (and in the special case of blobs; number of lobes and sharpness). The shapes are:
* **Multi-lobed blobs**: Jittered blobs consisting of multiple distinct lobes, ideal for simulating biological structures such as nuclei, cells, or other kinds of miscellaneous clumps of "stuff".
* **Toroids**: Jittered toroids that are helpful for augmenting if the goal is differentiating straight, tube like structures (vessels, axons).
