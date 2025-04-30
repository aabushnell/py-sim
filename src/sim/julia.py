import juliacall as jc
import numpy as np


# A0 = np.ndarray((1861,))
# L0 = np.ndarray((1861,))
# tau0 = np.ndarray((1861,1861))
# 
# io.read_array('data/A_0.csv', A0, 1861, True)
# io.read_array('data/L_0.csv', L0, 1861, True)
# io.read_matrix('data/tau_0.csv', tau0, 1861, True)
# 
# print('input data read')

# A0 = np.asarray([1, 1, 1])
# L0 = np.asarray([2, 1, 3])
# tau0 = np.asarray([[1, 2, 6], [2, 1, 3], [6, 3, 1]])

# theta = 8.0
# len = 1861
# 
# print('loading julia runtime...')

def calc_p_pi_julia(
    n_nodes: int,
    tau: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    P: np.ndarray,
    Pi: np.ndarray,
    coeff_theta: float
) -> None:

    julia_runtime = jc.newmodule('Main')
    jc.Pkg.activate('SolveInitial') # type: ignore
    julia_runtime.seval('using SolveInitial')

    result = julia_runtime.main(A, L, tau, coeff_theta, n_nodes)
    res_Pi = np.asarray(result[0])
    res_P = np.asarray(result[1])
    for i in range(n_nodes):
        P[i] = res_P[i]
        Pi[i] = res_Pi[i]

# io.write_array('data/log_Pi.csv', Pi, len)
# io.write_array('data/log_P.csv', P, len)

