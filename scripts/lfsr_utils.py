import sys
import numpy as np
from timeit import default_timer as timer
from typing import List
from accelerate.cuda.blas.api import Blas

blas = Blas()


def random_matrix(rows: int, cols: int):
    return np.random.choice([1, 0], size=(rows, cols), p=[.5, .5])


def throw_args_error(msg: str):
    print(msg)
    print("For arguments: ", sys.argv)
    sys.exit()


def build_matrix(elems: List[int], size: int):
    result = []
    iterator = elems.__iter__()

    for row in range(0, size):
        row = []
        for col in range(0, size):
            row.append(iterator.__next__())
        result.append(row)

    return np.matrix(result)


def parse_matrix_elems(args: str):
    result = []
    for x in str(args).split(","):
        result.append(int(x.strip()))
    return result


def create_init_state(size: int):
    initial = np.zeros((size, 1), dtype=int)
    initial[0] = 1
    return np.matrix(initial)


def cpu_multiply_mod_2(matrix_a, matrix_b):
    return np.mod(np.dot(matrix_a, matrix_b), 2)


def gpu_multiply_mod_2(matrix_a, matrix_b, size):
    blas.gemm(transa='T', transb='N', m=size, n=1, k=size, alpha=1, A=matrix_a, B=matrix_b, beta=0, C=matrix_b)
    """ C = alpha * transa(A) * transb(B) + beta * C """
    """ A of shape (m x k) """
    """ B of shape (k x n) """
    """ C of shape (m x n) """

    result = np.mod(matrix_b, 2)
    result = np.matrix(result, dtype=int)
    return result


def append_to_matrix(matrix, element):
    return np.append(matrix, element.flatten(), axis=0)


def matrix_contains(matrix, elem):
    return any(np.equal(matrix, elem.flatten()).all(1))


def run_with_timer(msg, function, *args):
    start = timer()
    result = function(*args)
    end = timer()
    print("\n{0} {1:f} s".format(msg, end - start))
    return result
