from mdapy import spline
import numpy as np


def test_SP():
    x = np.array([*np.linspace(0, 3, 50), *np.linspace(3.1, 5, 80)])

    y = np.sin(x)
    sp = spline.Spline(x, y)
    assert np.allclose(sp.evaluate(3.2), np.sin(3.2)), (
        "evaluate for single number is wrong."
    )
    assert np.allclose(sp.derivative(3.2), np.cos(3.2)), (
        "derivate for single number is wrong."
    )
    x1 = [1, 2, 3.5, 4.5]
    assert np.allclose(sp.evaluate(x1), np.sin(x1)), (
        "evaluate for multi number is wrong."
    )

    assert np.allclose(sp.derivative(x1), np.cos(x1)), (
        "derivate for multi number is wrong."
    )
