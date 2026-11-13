"""Minimal tests of the CT-HIN core on a toy batch (run with: python -m pytest tests)."""
import numpy as np

from ct_hin import ct_hin, entity_embedding, incidence, knn_graph, label_spreading


def toy_batch():
    # files 0-2 share a C2 address and a string; files 3-4 share utility strings; file 5 is isolated
    return [
        {"s": ["a", "b"], "ip": ["1.2.3.4"]}, {"s": ["a"], "ip": ["1.2.3.4"]}, {"s": ["b"], "ip": ["1.2.3.4"]},
        {"s": ["u1", "u2"]}, {"s": ["u1", "u2"]}, {"s": ["z"]},
    ]


def test_incidence_shape_and_filtering():
    X = incidence(toy_batch(), ["s", "ip"])
    assert X.shape == (6, 6)  # a, b, 1.2.3.4, u1, u2, z
    Z = entity_embedding(X, max_df=1.0)
    assert np.allclose(np.asarray(Z[5].sum()), 0)  # "z" occurs once and is dropped


def test_isolated_file_keeps_prior_and_cluster_is_smoothed():
    prior = np.array([3.0, 2.0, -1.0, -2.0, 1.0, 0.7])
    P, has_edge = knn_graph(entity_embedding(incidence(toy_batch(), ["s", "ip"]), max_df=1.0), k=2)
    f = label_spreading(P, has_edge, prior, alpha=0.9)
    assert not has_edge[5] and f[5] == prior[5]
    assert f[2] > 0.5 > prior[2]   # pulled up by its campaign neighbours
    assert f[4] < 0.5 < prior[4]   # pulled down by its benign twin


def test_wrapper_matches_components():
    prior = np.linspace(-2, 2, 6)
    a = ct_hin(toy_batch(), prior, kinds=["s", "ip"], max_df=1.0, k=2)
    P, h = knn_graph(entity_embedding(incidence(toy_batch(), ["s", "ip"]), max_df=1.0), k=2)
    assert np.allclose(a, label_spreading(P, h, prior, 0.9))
