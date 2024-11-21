"""
This package is a versatile deep-learning toolbox for PyTorch,
tailored to researchers working with N-dimensional vision problems,
and more specifically medial imaging problems.

It is intended to provide building blocks for a wide variety of
architectures, as well as a set of pre-defined backbones, as well as
a few task-specific models (segmentation, registration, synthesis, ...).

It will not provide domain-specific tools with dedicated pre- and post-
processing pipelines. However, such high-level tools can be implemented
using this toolbox.

Modules
-------
blending
    Task-specific models.
filters
    Task-agnostic architectures to use as backbones in models.
hist
    Building blocks for backbones and models.
shapes
    Differentiable functions to optimize during training.
texturizing
    Non-differentiable functions to compute during validation.
utils
    Tools to train networks.
"""
