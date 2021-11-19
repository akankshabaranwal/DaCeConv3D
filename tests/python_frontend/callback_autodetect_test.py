# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests automatic detection and baking of callbacks in the Python frontend. """
import dace
import numpy as np
import pytest

N = dace.symbol('N')


def dace_inhibitor(f):
    return f


@dace_inhibitor
def almost_gemm(A, alpha, B):
    return alpha * A @ B


@dace_inhibitor
def scale(C, beta):
    C *= beta


def test_automatic_callback():
    @dace.program
    def autocallback(A: dace.float64[N, N], B: dace.float64[N, N],
                     C: dace.float64[N, N], beta: dace.float64):
        tmp: dace.float64[N, N] = almost_gemm(A, 0.5, B)
        scale(C, beta)
        C += tmp

    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    C = np.random.rand(24, 24)
    beta = np.float64(np.random.rand())
    expected = 0.5 * A @ B + beta * C

    autocallback(A, B, C, beta)

    assert np.allclose(C, expected)


def test_automatic_callback_inference():
    @dace.program
    def autocallback_ret(A: dace.float64[N, N], B: dace.float64[N, N],
                         C: dace.float64[N, N], beta: dace.float64):
        tmp = np.ndarray([N, N], dace.float64)
        tmp[:] = almost_gemm(A, 0.5, B)
        scale(C, beta)
        C += tmp

    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    C = np.random.rand(24, 24)
    beta = np.float64(np.random.rand())
    expected = 0.5 * A @ B + beta * C

    autocallback_ret(A, B, C, beta)

    assert np.allclose(C, expected)


def test_automatic_callback_method():
    class NotDace:
        def __init__(self):
            self.q = np.random.rand()

        @dace_inhibitor
        def method(self, a):
            return a * self.q

    nd = NotDace()

    @dace.program
    def autocallback_method(A: dace.float64[N, N]):
        tmp: dace.float64[N, N] = nd.method(A)
        return tmp

    A = np.random.rand(24, 24)

    out = autocallback_method(A)

    assert np.allclose(out, nd.q * A)


@dace.program
def modcallback(A: dace.float64[N, N], B: dace.float64[N]):
    tmp: dace.float64[N] = np.median(A, axis=1)
    B[:] = tmp


def test_callback_from_module():
    N.set(24)
    A = np.random.rand(24, 24)
    B = np.random.rand(24)
    modcallback(A, B)
    diff = np.linalg.norm(B - np.median(A, axis=1))
    print('Difference:', diff)
    assert diff <= 1e-5


def sq(a):
    return a * a


@dace.program
def tasklet_callback(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a << A[i, j]
            b >> B[i, j]
            b = sq(a)


@pytest.mark.skip
def test_callback_tasklet():
    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    tasklet_callback(A, B)
    assert np.allclose(A * A, B)


def test_view_callback():
    @dace.program
    def autocallback(A: dace.float64[2 * N, N], B: dace.float64[N, N],
                     C: dace.float64[N, N], beta: dace.float64):
        A[N:, :] = almost_gemm(A[:N, :], 0.5, B)
        scale(C, beta)
        C += A[N:, :]

    A = np.random.rand(48, 24)
    B = np.random.rand(24, 24)
    C = np.random.rand(24, 24)
    beta = np.float64(np.random.rand())
    expected = 0.5 * A[:24] @ B + beta * C

    autocallback(A, B, C, beta)

    assert np.allclose(C, expected)


def test_print():
    @dace.program
    def printprog(a: dace.float64[2, 2]):
        print(a, 'hello')

    a = np.random.rand(2, 2)
    printprog(a)


if __name__ == '__main__':
    test_automatic_callback()
    # test_automatic_callback_inference()
    # test_automatic_callback_method()
    # test_callback_from_module()
    # test_view_callback()
    # test_callback_tasklet()
    test_print()
