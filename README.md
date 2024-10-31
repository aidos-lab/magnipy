# magnipy: Metric Space Magnitude Computations

This is a package for computing the **magnitude of a metric space**, which encodes the **effective size, diversity and geometry** of a metric space. Given a dataset or distance matrix, magnitude measures the effective number of distinct points in the space at a scale of dissimilarity between observations.

`Magnipy` enables the computation of:
- magnitude, **magnitude weights and magnitude functions** across varying resolutions
- magnitude dimension profiles and the **magnitude dimension** to estimate **intrinsic dimensionality**
- an **automated scale-finding** procedure to find suitable evaluation scales
- **MagArea** the area under a magnitude function, a multi-scale measure of the **intrinsic diversity** of a space
- **MagDiff** the area between two magnitude functions to measure the **difference in diversity** between two spaces


# Dependencies

Dependencies are managed using `poetry.` To setup the environment,
please run `poetry install` from the main directory.

# Running magnipy

All main implementations for computing magnitude are collected in the `Magnipy` class.
`tutorial_magnipy.ipynb` demonstrates how to set up and use this class to compute magnitude functions and magnitude dimension profiles.

# Citation
Please consider citing our work!

```bibtex
@misc{limbeck2023metric,
  title         = {Metric Space Magnitude for Evaluating the Diversity of Latent Representations}, 
  author        = {Katharina Limbeck and Rayna Andreeva and Rik Sarkar and Bastian Rieck},
  year          = {2023},
  eprint        = {2311.16054},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@inproceedings{andreeva2023metric,
  title         = {Metric Space Magnitude and Generalisation in Neural Networks},
  author        = {Andreeva, Rayna and Limbeck, Katharina and Rieck, Bastian and Sarkar, Rik},
  year          = 2023,
  booktitle     = {Proceedings of 2nd Annual Workshop on Topology, Algebra, and Geometry in Machine Learning~(TAG-ML)},
  volume        = 221,
  pages         = {242--253}
}
```
