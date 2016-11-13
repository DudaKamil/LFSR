import sys
import numpy as np

import scripts.lfsr_utils as utils
import scripts.lfsr_cpu as lfsr_cpu
import scripts.lfsr_gpu as lfsr_gpu
from typing import List

matrix = np.matrix(None)
desired_num_of_states = 0
size = 0


def process_arguments(args: List[str]):
    global matrix, desired_num_of_states, size
    matrix_elems = np.matrix(None)
    if len(args) == 2:
        size = int(args[1])
        matrix = utils.random_matrix(size, size)
    elif len(args) == 3:
        size = int(args[1])
        matrix_elems = utils.parse_matrix_elems(args[2])
        if len(matrix_elems) != size ** 2:
            utils.throw_args_error("Square matrix must be provided")
        matrix = utils.build_matrix(matrix_elems, size)
    else:
        utils.throw_args_error("Invalid number of arguments")
    desired_num_of_states = 2 ** size


process_arguments(sys.argv)
lfsr_cpu.do_check(matrix, size)
lfsr_gpu.do_check(matrix, size)
