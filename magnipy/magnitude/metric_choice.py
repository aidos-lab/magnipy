"Methods for lifting graphs to metric spaces."

import numpy as np
import networkx as nx
import sklearn
#from mag_edge_pool.src.metrics import standard_feature_metrics
from magnipy.magnitude.distances import get_dist
import magnipy.magnitude.metrics as supported_metrics

#  ╭──────────────────────────────────────────────────────────╮
#  │ Choosing the right Graph Metric                          │
#  ╰──────────────────────────────────────────────────────────╯

def choose_graph_metric(metric, mode="structure"):
    magnipy_metrics = [
        "Lp",
        "isomap",
        "torch_cdist",
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "euclidean",
        "hamming",
        "jaccard",
        "jensenshannon",
        "kulczynski1",
        "mahalanobis",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "yule",
    ]

    if mode == "attributes":
        if metric in magnipy_metrics:

            def get_metric(G):
                X = np.array([G.nodes[i]["feature"] for i in G.nodes])
                return get_dist(X, metric=metric)

        else:
            raise NotImplementedError("This metric is not implemented yet")

    elif mode == "structure":
        if metric in magnipy_metrics:

            def get_metric(G):
                Adj = nx.to_numpy_array(G)
                return get_dist(X=None, Adj=Adj, metric=metric)

        else:

            def get_metric(G):
                # G = nx.from_numpy_array(Adj)
                return lift_graph(G=G, metric=metric)

    elif mode == "full":
        if metric in magnipy_metrics:

            def get_metric(G):
                Adj = nx.to_numpy_array(G)
                X = np.array([G.nodes[i]["feature"] for i in G.nodes])
                return get_dist(X=X, Adj=Adj, metric=metric)

        else:
            raise NotImplementedError("This metric is not implemented yet")

    else:
        raise ValueError(
            "mode must be one of 'attributes', 'structure', or 'full'"
        )
    return get_metric


def to_nx_graph(x, adj):
    G = nx.from_numpy_array(adj)

    for i, feature in enumerate(x):
        G.nodes[i]["feature"] = feature
    return G



#  ╭──────────────────────────────────────────────────────────╮
#  │ Lifters                                                  │
#  ╰──────────────────────────────────────────────────────────╯

def lift_graph(G, metric, **kwargs):
    """Lift graph to a metric space.

    Lift graph to a metric space by calculating distances between its
    nodes using a specific graph metric.
    """

    # Immediately return zero metric space if graph is empty.
    if G.number_of_edges() == 0:
        n = G.number_of_nodes()
        return np.zeros((n, n))

    operator = getattr(supported_metrics, metric, None)

    if operator is None:
        raise RuntimeError(f"Unsupported metric: {metric}")

    return operator(G, **kwargs)
