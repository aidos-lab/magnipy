# magnipy: Metric Space Magnitude Computations

This is a repository for computing the **_magnitude of a metric space_**, which encodes the **effective size, diversity, and geometry** of a metric space. Given a dataset or distance matrix, **_magnitude_** measures the **effective number of distinct points** in the space at a scale of dissimilarity between observations.
We introduce the following codebase to compute and compare the magnitude of metric spaces.

<p align="center">
<img src="assets/magnipy_logo.svg" alt="magnipy logo"  style="float: right; width: 200px; height: 200px; ">
</p>

## 🔍 Main Functionalities and Classes

### `Magnipy`: For in-depth magnitude computations on a single metric space.

The functionalities of `Magnipy` for an individual metric space include:  
- Computing the metric space's **distance matrix** based on a feature space (attributes), adjecencies between observations (structure), or both features and adjecencies (full).
- Calculating the **similarity matrix** from the distances
- Executing an **automated scale-finding** procedure to find suitable evaluation scales
- Computing the **magnitude weight** of each point across multiple distance scales
- Evaluating and plotting **magnitude functions** across varying distance scales
- Estimating magnitude dimension profiles and calculating the **magnitude dimension** to quantify **intrinsic dimensionality**

### `Diversipy`: For comparing magnitude (and thus diversity) across multiple datasets.

The functionalities of `Diversipy` for a list of spaces (that share the same distance metric) include: 
- Executing an **automated scale-finding** procedure and the determining a **common evaluation interval across datasets**
- Computing **magnitude functions** across varying distance scales
- Calculating **MagArea**, the area under a magnitude function, a multi-scale measure of the **intrinsic diversity** of a dataset
- Calculating **MagDiff**, the area between magnitude functions, to measure the **difference in diversity** between datasets

### `Graphipy`: For in-depth magnitude computations on a single metric graph.

The functionalities of `Graphipy` for an individual metric graph include:  
- Calculating the magnitude of a `networkx` graph by computing magnitude seperately for each disconnected subgraph
- Computing the metric based of features (attributes), graph structure (structure), or both features and graph structure (full).
- Other functionalities similar to `Magnipy`


## ⚙️ Dependencies

To get started,
1. Clone this repository locally into a directory of your choosing
2. Create and activate a new Python virtual environment

Our dependencies are managed using the [`poetry`](https://python-poetry.org) package manager. Using your activated virtual environment, run the following to install `poetry`:

```python
$ pip install poetry
```

With `poetry` installed, run the following command from the main directory to download the necessary dependencies:

```python
$ poetry install
```

## 📚 Tutorials

Tutorials demonstrating the main functionalities can be found under the `notebooks` folder.

The following tutorials demonstrate how to initialize and utiltize the corresponding classes:
- `magnipy_tutorial.ipynb`:  Using `Magnipy` for computing the magnitude of one metric space
- `diversipy_tutorial.ipynb`: Using `Diversipy` for comparing the magnitude of multiple metric spaces
- `magnipy_tutorial.ipynb`:  Using `Graphipy` for computing the magnitude of metric graphs

The following supplementary demos are also provided:
- `mode_dropping.ipynb`: Using `MagDiff` for detecting mode dropping / mode collapse


## 📝 Citation
Please consider citing our work!

```bibtex
@inproceedings{limbeck2025geometry,
  title         = {Geometry-aware Edge Pooling for Graph Neural Networks}, 
  author        = {Katharina Limbeck and Lydia Mezrag and Guy Wolf and Bastian Rieck},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  year          = {2025 (in press)}
}

@inproceedings{limbeck2024metric,
  title         = {Metric Space Magnitude for Evaluating the Diversity of Latent Representations}, 
  author        = {Katharina Limbeck and Rayna Andreeva and Rik Sarkar and Bastian Rieck},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {37},
  year          = {2024}
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
