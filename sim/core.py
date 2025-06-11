from . import julia

import ctypes as C
# from ctypes.util import find_library
import sys

import numpy as np

###
# Load c++/CUDA shared libraries
###

# model simulation c++ library
# libsims_path = find_library('libsims')
# if not libsims_path:
#     print('ERROR: unable to find path to libsims library')
#     sys.exit()
try:
    libsims = C.CDLL('./sim/lib/libsims.so', mode=C.RTLD_GLOBAL)
except Exception as e:
    print('ERROR: unable to load libsims.so library')
    print(e)
    sys.exit()
print('libsims library successfully loaded')
print(libsims)

# CUDA matrix calculation library
# libcudmat_path = find_library('libcudmat')
# if not libcudmat_path:
#     print('ERROR: unable to find path to libcudmat library')
#     sys.exit()
try:
    libcudmat = C.CDLL('./sim/lib/libcudmat.so', mode=C.RTLD_GLOBAL)
except Exception as e:
    print('ERROR: unable to load libcudmat library')
    print(e)
    sys.exit()
print('libcudmat library successfully loaded')
print(libcudmat)

###
# libsims function bindings
###

alloc_model_raw = libsims.alloc_model       # fn alloc_model

alloc_model_raw.argtypes = [                # args:
    C.c_int                                 # int n_nodes
]
alloc_model_raw.restype = (                 # returns:
    C.POINTER(C.POINTER(C.c_double))        # double **
)

dealloc_model_raw = libsims.dealloc_model   # fn dealloc_model

dealloc_model_raw.argtypes = [              # args:
    C.POINTER(C.POINTER(C.c_double))        # double **model_ptrs
]
dealloc_model_raw.restype = (               # returns:
    None                                    # void
)

calc_p_pi_raw = libsims.calc_p_pi           # fn calc_p_pi

calc_p_pi_raw.argtypes = [                  # args:
    C.c_int,                                # int n_nodes
    C.POINTER(C.c_double),                  # double *tau
    C.POINTER(C.c_double),                  # double *A
    C.POINTER(C.c_double),                  # double *L
    C.POINTER(C.c_double),                  # double *P
    C.POINTER(C.c_double),                  # double *Pi
    C.c_double,                             # double coeff_theta
    C.c_double,                             # double diff_limit 
    C.c_int,                                # int debug_level   
    C.c_bool                                # bool relative_diff
]
calc_p_pi_raw.restype = (                   # returns:
    None                                    # void
)


calc_Y_raw = libsims.calc_Y                 # fn calc_Y

calc_Y_raw.argtypes = [                     # args
    C.c_int,                                # int n_nodes
    C.POINTER(C.c_double),                  # double *A
    C.POINTER(C.c_double),                  # double *L
    C.POINTER(C.c_double),                  # double *Y
    C.POINTER(C.c_double),                  # double *Pi
    C.c_double,                             # double coeff_theta
    C.c_int                                 # int debug_level
]
calc_Y_raw.restype = (                      # returns:
    None                                    # void
)

calc_X_raw = libsims.calc_X                 # fn calc_X

calc_X_raw.argtypes = [                     # args
    C.c_int,                                # int n_nodes
    C.POINTER(C.c_double),                  # double *tau
    C.POINTER(C.c_double),                  # double *X
    C.POINTER(C.c_double),                  # double *Y
    C.POINTER(C.c_double),                  # double *P
    C.POINTER(C.c_double),                  # double *Pi
    C.c_double,                             # double coeff_theta
    C.c_int                                 # int debug_level
]
calc_X_raw.restype = (                      # returns:
    None                                    # void
)

calc_Xi_raw = libsims.calc_Xi               # fn calc_Xi

calc_Xi_raw.argtypes = [                    # args
    C.c_int,                                # int n_nodes
    C.POINTER(C.c_double),                  # double *t
    C.POINTER(C.c_double),                  # double *Xi
    C.POINTER(C.c_double),                  # double *P
    C.POINTER(C.c_double),                  # double *Pi
    C.c_double,                             # double coeff_theta
    C.c_int,                                # int debug_level
    C.c_bool                                # bool allow_internal_trade
]
calc_Xi_raw.restype = (                     # returns:
    None                                    # void
)

update_A_raw = libsims.update_A             # fn update_A

update_A_raw.argtypes = [                   # args
    C.c_int,                                # int n_nodes
    C.POINTER(C.c_double),                  # double *Xi
    C.POINTER(C.c_double),                  # double *A
    C.c_double,                             # double coeff_eta
    C.c_double,                             # double coeff_beta
    C.c_double,                             # double coeff_sigma
    C.c_int,                                # int debug_level
    C.c_bool,                               # bool normalized
    C.c_bool,                               # bool log_A 
    C.c_bool                                # bool translate_A 
]
update_A_raw.restype = (                    # returns:
    None                                    # void
)

update_L_raw = libsims.update_t             # fn update_L

update_L_raw.argtypes = [                   # args
    C.c_int,                                # int n_nodes
    C.POINTER(C.c_double),                  # double *L
    C.POINTER(C.c_double),                  # double *B
    C.POINTER(C.c_double),                  # double *Y
    C.POINTER(C.c_double),                  # double *P
    C.c_double,                             # double coeff_a
    C.c_double,                             # double coeff_b
    C.c_double,                             # double coeff_f
    C.c_double,                             # double coeff_d
    C.c_double,                             # double coeff_xi
    C.c_double,                             # double coeff_lambda
    C.c_int                                 # int debug_level
]
update_L_raw.restype = (                    # returns:
    None                                    # void
)

update_t_raw = libsims.update_t             # fn update_t

update_t_raw.argtypes = [                   # args
    C.c_int,                                # int n_nodes
    C.POINTER(C.c_double),                  # double *t
    C.POINTER(C.c_double),                  # double *Xi
    C.POINTER(C.c_double),                  # double *Xi_prev
    C.c_double,                             # double coeff_chi
    C.c_int                                 # int debug_level
]
update_L_raw.restype = (                    # returns:
    None                                    # void
)

###
# libcudmat function bindings
###

calc_tau_raw = libcudmat.calc_tau_cublas    # fn calc_tau_cublas:

calc_tau_raw.argtypes = [                   # args:
    C.c_int,                                # int n_nodes 
    C.POINTER(C.c_double),                  # const double *t 
    C.POINTER(C.c_double),                  # double *tau 
    C.c_double,                             # double coeff_theta
    C.c_int,                                # int n_iterations
    C.c_int                                 # int debug_level
]
calc_tau_raw.restype = (                    # returns:
    None                                    # void
)

class Model:
    # positions of array pointers within overall model_ptrs array
    ptr_indices = ['t', 'tau', 'X', 'Xi', 'Xi_prev', 'Xi_temp','A', 'L', 'B', 'Y', 'P', 'Pi']

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        # allocate memory and return pointers for model
        self.model_ptrs = alloc_model_raw(C.c_int(n_nodes))
        # link model pointers to numpy arrays
        self.t = np.ctypeslib.as_array(self.model_ptrs[0], shape=(n_nodes, n_nodes))
        self.tau = np.ctypeslib.as_array(self.model_ptrs[1], shape=(n_nodes, n_nodes))
        self.X = np.ctypeslib.as_array(self.model_ptrs[2], shape=(n_nodes, n_nodes))
        self.Xi = np.ctypeslib.as_array(self.model_ptrs[3], shape=(n_nodes, n_nodes))
        self.A = np.ctypeslib.as_array(self.model_ptrs[6], shape=(n_nodes,))
        self.L = np.ctypeslib.as_array(self.model_ptrs[7], shape=(n_nodes,))
        self.B = np.ctypeslib.as_array(self.model_ptrs[8], shape=(n_nodes,))
        self.Y = np.ctypeslib.as_array(self.model_ptrs[9], shape=(n_nodes,))
        self.P = np.ctypeslib.as_array(self.model_ptrs[10], shape=(n_nodes,))
        self.Pi = np.ctypeslib.as_array(self.model_ptrs[11], shape=(n_nodes,))

        self._Xi_prev = np.ctypeslib.as_array(self.model_ptrs[4], shape=(n_nodes, n_nodes))
        self._Xi_temp = np.ctypeslib.as_array(self.model_ptrs[5], shape=(n_nodes, n_nodes))
        # internal
        self.epoch = 0

    def get_raw_ptr(self, array_name: str):
        if array_name not in self.ptr_indices:
            print(f'ERROR: array {array_name} is not a valid part of the model')
            return
        return self.model_ptrs[self.ptr_indices.index(array_name)]

    def calc_tau(
        self, 
        coeff_theta: 
        float, 
        n_iterations: 
        int, 
        debug_level: int = 0
    ):
        calc_tau_raw(
            C.c_int(self.n_nodes),
            self.get_raw_ptr('t'),
            self.get_raw_ptr('tau'),
            C.c_double(coeff_theta),
            C.c_int(n_iterations), 
            C.c_int(debug_level)
        )

    def init_p_pi(
        self, 
        coeff_theta: float
    ):
        julia.calc_p_pi_julia(
            self.n_nodes,
            self.tau,
            self.A,
            self.L,
            self.P,
            self.Pi,
            coeff_theta
        )

    def calc_p_pi(
        self, 
        coeff_theta: float, 
        diff_limit: float, 
        debug_level: int, 
        relative_diff: bool = False
    ):
        calc_p_pi_raw(
            C.c_int(self.n_nodes), 
            self.get_raw_ptr('tau'), 
            self.get_raw_ptr('A'), 
            self.get_raw_ptr('L'), 
            self.get_raw_ptr('P'), 
            self.get_raw_ptr('Pi'), 
            C.c_double(coeff_theta), 
            C.c_double(diff_limit),
            C.c_int(debug_level),
            C.c_bool(relative_diff)
        )

    def calc_Y(
        self, 
        coeff_theta: float, 
        debug_level: int
    ):
        calc_Y_raw(
            C.c_int(self.n_nodes),
            self.get_raw_ptr('A'),
            self.get_raw_ptr('L'),
            self.get_raw_ptr('Y'),
            self.get_raw_ptr('Pi'),
            C.c_double(coeff_theta),
            C.c_int(debug_level)
        )

    def calc_X(
        self, 
        coeff_theta: float, 
        debug_level: int
    ):
        calc_X_raw(
            C.c_int(self.n_nodes),
            self.get_raw_ptr('tau'),
            self.get_raw_ptr('X'),
            self.get_raw_ptr('Y'),
            self.get_raw_ptr('P'),
            self.get_raw_ptr('Pi'),
            C.c_double(coeff_theta),
            C.c_int(debug_level)
        )

    def calc_Xi(self, coeff_theta: float, debug_level: int):
        calc_Xi_raw(
            C.c_int(self.n_nodes),
            self.get_raw_ptr('t'),
            self.get_raw_ptr('Xi'),
            self.get_raw_ptr('P'),
            self.get_raw_ptr('Pi'),
            C.c_double(coeff_theta),
            C.c_int(debug_level)
        )

    def update_A(self, coeff_eta: float, coeff_beta: float, 
                 coeff_sigma: float, debug_level: int, 
                 normalized: bool = False, 
                 log_A: bool = True, 
                 translate_A: bool = True):
        update_A_raw(
            C.c_int(self.n_nodes),
            self.get_raw_ptr('Xi'),
            self.get_raw_ptr('A'),
            C.c_double(coeff_eta),
            C.c_double(coeff_beta),
            C.c_double(coeff_sigma),
            C.c_int(debug_level),
            C.c_bool(normalized),
            C.c_bool(log_A),
            C.c_bool(translate_A)
        )

    def update_L(self, coeff_a: float, coeff_b: float, 
                 coeff_f: float, coeff_d: float, coeff_xi: float, 
                 coeff_lambda: float, debug_level: int):
        update_L_raw(
            C.c_int(self.n_nodes),
            self.get_raw_ptr('L'),
            self.get_raw_ptr('B'),
            self.get_raw_ptr('Y'),
            self.get_raw_ptr('P'),
            C.c_double(coeff_a),
            C.c_double(coeff_b),
            C.c_double(coeff_f),
            C.c_double(coeff_d),
            C.c_double(coeff_xi),
            C.c_double(coeff_lambda),
            C.c_int(debug_level)
        )

    # ** incomplete implementation **
    def update_t(self, coeff_chi: float, debug_level: int):
        for i in range(self.n_nodes):
            self._Xi_temp[i] = self.Xi[i]
        update_t_raw(
            C.c_int(self.n_nodes),
            self.get_raw_ptr('t'),
            self.get_raw_ptr('Xi'),
            self.get_raw_ptr('Xi_prev'),
            C.c_double(coeff_chi),
            C.c_int(debug_level)
        )
        for i in range(self.n_nodes):
            self._Xi_prev[i] = self._Xi_temp[i]

###
# Wrapper functions
###




