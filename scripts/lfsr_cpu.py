import numpy as np

import scripts.lfsr_utils as utils

initial_state = np.matrix(None)
states = []


def check_lfsr_cpu(matrix, size):
    global initial_state, states
    initial_state = utils.create_init_state(len(matrix))
    desired_num_of_states = 2 ** size

    current_state = initial_state
    zeros = np.zeros((1, len(matrix)), dtype=int)
    states = utils.append_to_matrix(zeros, current_state)

    while True:
        current_state = utils.cpu_multiply_mod_2(matrix, current_state)
        if utils.matrix_contains(states, current_state):
            break
        states = utils.append_to_matrix(states, current_state)

    return len(states) == desired_num_of_states


def do_check(matrix, size):
    result = utils.run_with_timer("CPU time:", check_lfsr_cpu, matrix, size)
    print("Result: ", result)
    print("States: ", len(states))
