"""
SynthShapes
===========

SynthShapes is a Python library for generating synthetic shapes and applying
filters for image augmentation.

Modules
-------
- blending: Functions for combining shapes.
- filters: Tools for applying intensity-based transformations.
- shapes: Classes for generating geometric and multi-lobed shapes.
- utils: Utility functions for saving/loading shapes.

Examples
--------
>>> from SynthShapes import generate_shape, blend_shapes
>>> shape = generate_shape('sphere', radius=5)
>>> blended = blend_shapes(shape, other_shape)
"""
from . import blending, filters, hist, shapes, texturizing, utils

__all__ = ["blending", "filters", "hist", "shapes", "texturizing", "utils"]
