import numpy as np


def test_subpackages_import():
    import fabric  # noqa: F401
    import fabric.data  # noqa: F401
    import fabric.eval  # noqa: F401
    import fabric.models  # noqa: F401
    import fabric.train  # noqa: F401
    import fabric.utils  # noqa: F401


def test_tiny_graph_laplacian():
    # 4-cycle graph: eigenvalues of its Laplacian are {0, 2, 2, 4}.
    A = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=float,
    )
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigs = np.sort(np.linalg.eigvalsh(L))
    expected = np.array([0.0, 2.0, 2.0, 4.0])
    np.testing.assert_allclose(eigs, expected, atol=1e-9)
