import numpy as np
from accelerate.cuda.blas.api import Blas
import scripts.lfsr_utils as utils

blas = Blas()

# a = np.matrix([[0, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
# b = np.matrix([[1], [0], [0]], dtype=np.float64)

a = np.matrix([[1, 1, 1], [0, 0, 0], [0, 1, 0]], dtype=np.float64)
a = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
b = np.matrix([[1], [2], [4]], dtype=np.float64)

cpu_result = utils.cpu_multiply_mod_2(np.matrix(a, dtype=np.int), np.matrix(b, dtype=np.int))
gpu_result = utils.gpu_multiply_mod_2(a, b, len(a))

print("CPU\n", cpu_result)
print("GPU\n", gpu_result)
