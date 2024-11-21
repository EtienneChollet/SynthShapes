# SynthShapes
SynthShapes is a PyTorch toolbox for generating synthetic 3D shapes, tailored specifically for augmenting biomedical imaging datasets. It allows users to create complex, realistic shapes for use in machine learning pipelines, especially when training data is scarce or needs augmentation.

# Features
* Generate a variety of 3D shapes (e.g., blobs, toroids).
* Easily customizable parameters (size, shape, intensity).
* Ready-to-go augmentation modules for biomedical image datasets (real or synth).

## Available Shapes
Although there are currently only two available shapes to choose from, they span a wide range of configurations, thanks to their jitter (and in the special case of blobs; number of lobes and sharpness). The shapes are:

* **Multi-lobed blobs**: Jittered blobs consisting of multiple distinct lobes, ideal for simulating biological structures such as nuclei, cells, or other kinds of miscellaneous clumps of "stuff".
* **Toroids**: Jittered toroids that are helpful for augmenting if the goal is differentiating straight, tube like structures (vessels, axons).
