import numpy as np

###
# Utility functions
###

def read_array(filepath: str, array: np.ndarray, array_len: int, indexed: bool = False):
    with open(filepath, 'r') as file:
        for i in range(array_len):
            if indexed:
                array[i] = float(file.readline().split(sep=',')[1])
            else:
                array[i] = float(file.readline())


def read_matrix(filepath: str, matrix: np.ndarray, array_len: int, flat: bool = True):
    with open(filepath, 'r') as file:
        lines = [line.rstrip() for line in file]
        if flat:
            for line in lines:
                i, j, val = line.split(sep=',')
                i, j, val = int(i), int(j), float(val)
                if i >= 0 and j >= 0:
                    matrix[int(i)][int(j)] = float(val)
        else:
            print('ERROR: not implemented!')
            raise NotImplementedError


def write_array(filepath: str, array: np.ndarray, array_len: int):
    with open(filepath, 'w') as file:
        for i in range(array_len):
            print(array[i], file=file)

