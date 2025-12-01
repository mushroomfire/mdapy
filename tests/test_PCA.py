from sklearn.decomposition import PCA as PCA_S
from mdapy.potential_tool import PCA
import numpy as np


def test_pca():
    np.random.seed(1)
    des = np.random.random((100, 20))

    pca1 = PCA_S(n_components=3)
    res1 = pca1.fit_transform(des)

    pca2 = PCA(n_components=3)
    res2 = pca2.fit_transform(des)

    assert np.allclose(res1, res2), "pca res is wrong"
    assert np.allclose(pca1.explained_variance_, pca2.explained_variance), (
        "pca explained_variance is wrong."
    )
