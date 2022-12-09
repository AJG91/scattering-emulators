"""
Created on Oct 18, 2013

@author: Kyle Wendt
"""

from fastcache import lru_cache

import numpy
from numpy import arange, array, diag, ones_like, sqrt, stack, zeros
from numpy.linalg import eigh, eigvalsh

from enum import IntEnum, unique as uniqueEnum


@uniqueEnum
class QuadratureType(IntEnum):
    GaussLegendre = 0
    GaussLabotto = 1
    SemiInfinite = 2
    ExponentialGaussLegendre = 3
    ExponentialGaussLabotto = 4

    @classmethod
    def from_suffix(cls, suffix):
        suffix = str(suffix).lower()
        if suffix == "l":
            return cls.GaussLabotto
        if suffix == "i":
            return cls.SemiInfinite
        if suffix == "e":
            return cls.ExponentialGaussLegendre
        if suffix == "f":
            return cls.ExponentialGaussLabotto
        return cls.GaussLegendre


@lru_cache(maxsize=1024)
def gauss_lobatto_mesh(n):
    if n < 2:
        raise ValueError("n must be > 1")
    if n == 2:
        xi = array((-1.0, +1.0))
        wi = array((+1.0, +1.0))
        return stack((xi, wi))

    xi = zeros(n)
    wi = zeros(n)
    Pn = zeros(n)
    i = arange(1, n - 2)
    b = sqrt(
        (i * (2.0 + i)) / (3.0 + 4.0 * i * (2.0 + i))
    )  # coeff for Jacobi Poly with a=b=1

    M = diag(b, -1) + diag(b, 1)
    xi[1 : n - 1] = eigvalsh(M)
    xi[0] = -1.0
    xi[-1] = 1.0

    Pim2 = ones_like(xi)  # P_{i-2}
    Pim1 = xi  # P_{i-1}
    for j in range(2, n):  # want P_{n-1}
        wi = (1.0 / j) * ((2 * j - 1) * xi * Pim1 - (j - 1) * Pim2)
        Pim2 = Pim1
        Pim1 = wi
    wi = 2.0 / (n * (n - 1) * wi ** 2)
    wi[0] = wi[-1] = 2.0 / (n * (n - 1))
    return stack((xi, wi))


@lru_cache(maxsize=1024)
def gauss_legendre_mesh(n):
    if n < 2:
        raise ValueError("n must be > 1")
    if n == 2:
        xi = array((-0.5773502691896257, +0.5773502691896257))
        wi = array((+1.0, +1.0))
        return stack((xi, wi))

    Pn = zeros(n)
    i = arange(1, n)
    b = i / sqrt((2.0 * i - 1) * (2.0 * i + 1))

    M = diag(b, -1) + diag(b, 1)
    xi, Wi = eigh(M)
    return stack((xi, 2 * Wi[0, :] ** 2))


def _gll_mesh(nodes, num_points):
    """
    Construct a compound Gauss-Legendre or Gauss-Lobatto-Legendre integration
    quadrature.  Adjacent Gauss-Lobatto-Legendre sub quadratures will have a
    shared mesh point, therefore the total number of mesh points may be less
    than sum(num_points).

    :param nodes:  List of nodes define each sub quadrature
    :param num_points: List of number of points in each sub quadrature
    :return: quadrature node, quadurature weights
    """
    if nodes.size != num_points.size + 1:
        raise ValueError("len(nodes) != len(num_points) + 1")

    nn = len(num_points)
    kind = zeros(num_points.size, numpy.int64)
    for i in range(nn):
        if num_points[i] < 0:
            kind[i] = 1
            num_points[i] = -num_points[i]

    nt = num_points[0]
    for i in range(1, nn):
        N = num_points[i]
        T = kind[i]
        nt += N
        if kind[i] == 1 and kind[i - 1] == 1:
            nt -= 1

    xi = zeros(nt, numpy.float64)
    wi = zeros(nt, numpy.float64)
    o = 0
    prev_k = 0
    for i, (n, k) in enumerate(zip(num_points, kind)):
        A = nodes[i]
        B = nodes[i + 1]
        if k == 1:
            XW = gauss_lobatto_mesh(n)
        else:
            XW = gauss_legendre_mesh(n)

        X = XW[0] * (B - A) / 2.0 + (A + B) / 2
        W = XW[1] * (B - A) / 2.0

        if k == 1 and prev_k == 1:
            n -= 1
            wi[o - 1] += W[0]
            X = X[1:]
            W = W[1:]
        prev_k = k

        xi[o : o + n] = X
        wi[o : o + n] = W
        o += n
    return stack((xi, wi))


def gll_mesh(nodes, num_points):
    """
    Construct a compound Gauss-Legendre or Gauss-Lobatto-Legendre integration
    quadrature.  Adjacent Gauss-Lobatto-Legendre sub quadratures will have a
    shared mesh point, therefore the total number of mesh points may be less
    than sum(num_points).

    :param nodes:  List of nodes define each sub quadrature
    :param num_points: List of number of points in each sub quadrature
    :return: quadrature node, quadurature weights
    """
    from numpy import array, ndarray
    import re

    rec = re.compile(r"(?P<n>\d+)(?P<t>[lg]?)")

    def to_int(n):
        n, t = rec.match(n.lower()).groups("g")
        if t == "l":
            return -abs(int(n))
        return abs(int(n))

    if isinstance(nodes, str):
        nodes = tuple(map(float, nodes.replace(",", " ").split()))
    nodes = array(nodes, float).ravel()

    if isinstance(num_points, str):
        num_points = tuple(map(to_int, num_points.lower().split()))
    num_points = array(num_points).ravel()
    return _gll_mesh(nodes, num_points)


@lru_cache(maxsize=1024)
def Exponential_Mesh(x0, xf, npts, terrible=False):
    from numpy import arange, log

    if not terrible:
        n, ni = gll_mesh("0 1", "{:d}L".format(npts))
    else:
        n = arange(npts) / (npts - 1)
        ni = 1.0 / (npts - 1)
    s, f = float(min(x0, xf)), float(max(x0, xf))
    assert s > 0, "smaller scale must be greater than 0"
    return s * (f / s) ** n, s * log(f / s) * (f / s) ** n * ni


from numpy import arange, array, log


# this function can be used in numba with nopython=True
def _compound_mesh(nodes, num_points, kind):
    if (
        nodes.size == 2
        and num_points.size == 1
        and kind[0] == QuadratureType.SemiInfinite
    ):
        start = nodes[0]
        scale = nodes[1]
        XW = gauss_legendre_mesh(num_points[0])
        xi = start + scale * (1.0 + XW[0]) / (1.0 - XW[0])
        wi = 2.0 * scale * XW[1] / (1 - XW[0]) ** 2
        return stack((xi, wi))

    if num_points.size != kind.size:
        raise ValueError("len(num_points) != len(kind)")
    if nodes.size != num_points.size + 1 and nodes.size != num_points.size:
        raise ValueError("len(nodes) - len(num_points) != 0 or 1 ")

    nt = num_points[0]
    for i in range(1, num_points.size):
        N = num_points[i]
        nt += N
        if kind[i] == kind[i - 1] == QuadratureType.GaussLabotto:
            nt -= 1

    xi = zeros(nt, numpy.float64)
    wi = zeros(nt, numpy.float64)
    o = 0
    prev_k = kind[0]
    for i, (n, k) in enumerate(zip(num_points, kind)):
        if k == QuadratureType.GaussLegendre or k == QuadratureType.GaussLabotto:
            A = nodes[i]
            B = nodes[i + 1]
            if k == QuadratureType.GaussLabotto:
                XW = gauss_lobatto_mesh(n)
            else:
                XW = gauss_legendre_mesh(n)
            X = XW[0] * (B - A) / 2.0 + (A + B) / 2
            W = XW[1] * (B - A) / 2.0
            if k == prev_k == QuadratureType.GaussLabotto:
                n -= 1
                wi[o - 1] += W[0]
                X = X[1:]
                W = W[1:]
            xi[o : o + n] = X
            wi[o : o + n] = W
            o += n
        if k == QuadratureType.SemiInfinite:
            if i != num_points.size - 1:
                raise ValueError("SemiInfinite only valid for last interval")
            scale = nodes[i]
            XW = gauss_legendre_mesh(n)
            X = scale * (1.0 + (1.0 + XW[0]) / (1.0 - XW[0]))
            W = 2.0 * scale * XW[1] / (1 - XW[0]) ** 2
            xi[o : o + n] = X
            wi[o : o + n] = W
            o += n
        if (
            k == QuadratureType.ExponentialGaussLegendre
            or k == QuadratureType.ExponentialGaussLabotto
        ):
            if k == QuadratureType.ExponentialGaussLabotto:
                XW = gauss_lobatto_mesh(n)
            else:
                XW = gauss_legendre_mesh(n)
            s = nodes[i]
            f = nodes[i + 1]
            X = s * (f / s) ** XW[0]
            W = s * log(f / s) * (f / s) ** XW[0] * XW[1]
            if k == prev_k == QuadratureType.GaussLabotto:
                n -= 1
                wi[o - 1] += W[0]
                X = X[1:]
                W = W[1:]
            xi[o : o + n] = X
            wi[o : o + n] = W
            o += n
        prev_k = k
    return stack((xi, wi))


def compound_mesh(node_spec, points_spec):
    """

    Parameters
    ----------
    node_spec :
        List of nodes define each sub quadrature
    points_spec :
        List of number of points in each sub quadrature

    Returns
    -------
    quadrature node, quadurature weights
    """
    from numpy import array, ndarray, zeros_like
    import re

    rec = re.compile(r"(?P<n>\d+)(?P<t>[glief]?)")

    def to_int(n):
        n, t = rec.match(n.lower()).groups()
        t = t or "g"
        return abs(int(n)), QuadratureType.from_suffix(t).value

    if isinstance(node_spec, str):
        node_spec = tuple(map(float, node_spec.replace(",", " ").split()))
    node_spec = array(node_spec, float).ravel()

    if isinstance(points_spec, str):
        points_spec, type_spec = array(
            tuple(map(to_int, points_spec.lower().split()))
        ).T
    else:
        points_spec = array(points_spec).ravel()
        type_spec = zeros_like(points_spec)
    if len(type_spec) == len(node_spec):
        type_spec[-1] = QuadratureType.SemiInfinite

    return _compound_mesh(node_spec, points_spec, type_spec)
