from sim.core import *
from sim.io import *

# standard array len
# represents number of nodes in model
SIZE = 1861

test_model = Model(SIZE)

read_array('data/A.csv', test_model.A, SIZE)
read_array('data/L.csv', test_model.L, SIZE)
read_array('data/B.csv', test_model.B, SIZE)

read_matrix('data/t.csv', test_model.t, SIZE)

coeff_theta = 2.0

print('calculating tau...')

test_model.calc_tau(coeff_theta, 60, 1)

# write_matrix('data/tau.csv', test_model.tau, SIZE)

print('initializing p/pi...')

test_model.init_p_pi(coeff_theta)

test_model.calc_p_pi(coeff_theta, float(pow(10, -3)), 0)

