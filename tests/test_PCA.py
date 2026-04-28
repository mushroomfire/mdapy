# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""PCA — fixture-driven, no sklearn at runtime."""

import numpy as np

from mdapy.potential_tool import PCA
from _fixture_helper import load_advanced


def test_pca():
    data = load_advanced("pca")
    np.random.seed(int(data["seed"]))
    des = np.random.random((int(data["n_samples"]), int(data["n_features"])))
    pca = PCA(n_components=int(data["n_components"]))
    res = pca.fit_transform(des)

    assert np.allclose(res, data["transformed"]), "PCA transformed coords differ"
    assert np.allclose(pca.explained_variance, data["explained_variance"]), (
        "PCA explained_variance differs"
    )
