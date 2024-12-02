# magnipy: Metric Space Magnitude Computations

This is a repository for computing the **_magnitude of a metric space_**, which encodes the **effective size, diversity, and geometry** of a metric space.  

Given a dataset or distance matrix, **_magnitude_** measures the **effective number of distinct points** in the space at a scale of dissimilarity between observations.

This repository supports the NeurIPS 2024 paper: [Metric Space Magnitude for Evaluating the Diversity of Latent Representations](https://arxiv.org/abs/2311.16054).

## Dependencies

Dependencies are managed using the [`poetry`](https://python-poetry.org) package manager.

With poetry installed and an active virtual environment, run the following command from the main directory to download the necessary dependencies:

```python
$ poetry install
```

## Usage

We introduce two classes to aid in the computation and comparison of the magnitude of metric spaces:

### 1. `Diversipy`: For comparing magnitude (& thus diversity) across different metric spaces.

Core functionalities of `Diversipy` for a list of metric spaces include: 
- Executing an **automated scale-finding** procedure and the determining a **common evaluation interval across metric spaces**
- Computing **magnitude functions** across varying distance scales
- Calculating **MagArea**: the area under a magnitude function, a multi-scale measure of the **intrinsic diversity** of a space
- Calculating **MagDiff**: the area between two magnitude functions to measure the **difference in diversity** between two spaces


### 2. `Magnipy`: For in-depth magnitude computations on a single metric space.

Core functionalities of `Magnipy` for an individual metric space include:  
- Computing the metric space's **distance matrix**
- Creating the related **similarity matrix**
- Executing an **automated scale-finding** procedure to find a suitable evaluation scale
- Finding the **magnitude weight** of each point across distance scales
- Defining **magnitude functions** across varying resolutions
- Creation of magnitude dimension profiles and the **magnitude dimension** to estimate **intrinsic dimensionality**


## Tutorials

Separate tutorials for the `Diversipy` and `Magnipy` classes can be found under the `notebooks` folder (`diversipy_tutorial.ipynb` and `magnipy_tutorial.ipynb` respectively), and are aided by the `tutorial_data.py` script.

Each tutorial demonstrates how to initialize and utiltize its corresponding class to execute core functionalities.

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
  year          = {2023},
  booktitle     = {Proceedings of 2nd Annual Workshop on Topology, Algebra, and Geometry in Machine Learning~(TAG-ML)},
  volume        = {221},
  pages         = {242--253}
}
```
