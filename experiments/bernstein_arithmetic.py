"""Outward tensor-product Bernstein algebra on the unit square.

All combinatorial weights are enclosed against their exact rational values.
The verifier uses elementwise binary64 operations, never untracked BLAS sums.
"""
from functools import lru_cache
from fractions import Fraction as F
from math import comb

import numpy as np

import direct_residual_certificate as exact


def rational(value):
    value = F(value)
    nearest = float(value)
    if not np.isfinite(nearest):
        raise ArithmeticError("Nonfinite rational enclosure")
    if F(nearest) == value:
        return nearest, nearest
    lo, hi = float(exact.down(nearest)), float(exact.up(nearest))
    if not F(lo) <= value <= F(hi):
        raise ArithmeticError("Rational conversion did not enclose")
    return lo, hi


def constant(value):
    lo, hi = rational(value)
    return np.array([[lo]]), np.array([[hi]])


def times(a, b):
    lower = np.minimum(np.minimum(a[0]*b[0], a[0]*b[1]), np.minimum(a[1]*b[0], a[1]*b[1]))
    upper = np.maximum(np.maximum(a[0]*b[0], a[0]*b[1]), np.maximum(a[1]*b[0], a[1]*b[1]))
    lo, hi = exact.down(lower), exact.up(upper)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ArithmeticError("Nonfinite Bernstein arithmetic")
    return lo, hi


def scale(a, factor):
    if not factor:
        return constant(0)
    if factor == 1:
        return a
    if factor == -1:
        return -a[1], -a[0]
    return times(a, rational(factor))


@lru_cache(maxsize=None)
def product_weights(n, m):
    lower, upper = np.empty((n+1, m+1)), np.empty((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            lower[i, j], upper[i, j] = rational(F(comb(n, i)*comb(m, j), comb(n+m, i+j)))
    return lower, upper


def multiply(a, b):
    if (not np.any(a[0]) and not np.any(a[1])) or (not np.any(b[0]) and not np.any(b[1])):
        return constant(0)
    if a[0].size == 1 and a[0].item() == a[1].item() == 1:
        return b
    if b[0].size == 1 and b[0].item() == b[1].item() == 1:
        return a
    if a[0].size > b[0].size:
        a, b = b, a
    n, m = np.array(a[0].shape)-1, np.array(b[0].shape)-1
    ws, wy = product_weights(n[0], m[0]), product_weights(n[1], m[1])
    lower, upper = np.zeros(n+m+1), np.zeros(n+m+1)
    for i in range(n[0]+1):
        for j in range(n[1]+1):
            if a[0][i, j] == a[1][i, j] == 0:
                continue
            term = times((a[0][i, j], a[1][i, j]), b)
            term = times(term, (ws[0][i, :, None], ws[1][i, :, None]))
            term = times(term, (wy[0][j, None, :], wy[1][j, None, :]))
            target = np.s_[i:i+m[0]+1, j:j+m[1]+1]
            lower[target] = exact.down(lower[target]+term[0])
            upper[target] = exact.up(upper[target]+term[1])
    return lower, upper


def elevate(a, shape):
    n = np.array(shape)-np.array(a[0].shape)
    if (n < 0).any():
        raise ValueError("Degree reduction requested")
    if not n.any():
        return a
    ones = np.ones(n+1)
    return multiply((ones, ones), a)


def add(a, b):
    if not np.any(a[0]) and not np.any(a[1]):
        return b
    if not np.any(b[0]) and not np.any(b[1]):
        return a
    shape = np.maximum(a[0].shape, b[0].shape)
    a, b = elevate(a, shape), elevate(b, shape)
    return exact.down(a[0]+b[0]), exact.up(a[1]+b[1])


def sum_all(polys):
    answer = constant(0)
    for p in polys:
        answer = add(answer, p)
    return answer


def derivative(a, axis):
    n = a[0].shape[axis]-1
    if not n:
        return constant(0)
    lo, hi = (np.moveaxis(v, axis, 0) for v in a)
    result = (exact.down(lo[1:]-hi[:-1]), exact.up(hi[1:]-lo[:-1]))
    result = scale(result, F(n))
    return tuple(np.moveaxis(v, 0, axis) for v in result)


def rational_bernstein(poly):
    """Exact coefficients, retained until after coarse affine restriction."""
    ns = max((i for i, j in poly), default=0)
    ny = max((j for i, j in poly), default=0)
    first = [[F(0) for _ in range(ny+1)] for _ in range(ns+1)]
    for (i, j), value in poly.items():
        for k in range(i, ns+1):
            first[k][j] += value*F(comb(k, i), comb(ns, i))
    return np.array([[sum((first[k][j]*F(comb(l, j), comb(ny, j))
                          for j in range(l+1)), F(0))
                      for l in range(ny+1)] for k in range(ns+1)], dtype=object)


def rational_split(coefficients, axis):
    work = np.moveaxis(coefficients, axis, 0).copy()
    left, right = np.empty_like(work), np.empty_like(work)
    left[0], right[-1] = work[0], work[-1]
    for k in range(1, len(work)):
        work = (work[:-1]+work[1:])*F(1, 2)
        left[k], right[-1-k] = work[0], work[-1]
    return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)


def enclose_rationals(coefficients):
    lo, hi = np.empty(coefficients.shape), np.empty(coefficients.shape)
    for index in np.ndindex(coefficients.shape):
        lo[index], hi[index] = rational(coefficients[index])
    return lo, hi
