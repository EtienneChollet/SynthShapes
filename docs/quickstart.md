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
