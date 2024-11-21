"""
SynthShapes is a primitive yet powerful PyTorch-based toolbox designed to
generate synthetic 3D shapes, tailored specifically for augmenting biomedical
imaging datasets. It is built to address the challenges faced in machine
learning pipelines for biomedical imaging, such as limited availability of
annotated data, the need for diverse training datasets, and the necessity of
domain-specific augmentations.

Modules
-------
blending
    Tools for alpha-blending tensors.
filters
    Kernel-based spatial filters/transformations.
hist
    Modules to map intensity histograms to one another.
shapes
    Augmentation layers for incorporating synthetic shapes into 3D images.
texturizing
    Tools to apply textures to label maps.
utils
    General-purpose utilities for comon operations.
"""
