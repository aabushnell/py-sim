# LRG Model Simulation Code Documentation

Note that this is an initial draft of both the final version of the code and documentation. Any contents herein are subject to change.

## Table of contents
1. [Introduction](#introduction)
2. [Installation](#installation)
    1. [Requirements](#requirements)
    2. [Walkthrough](#walkthrough)
3. [Usage](#usage)
    1. [Initialization](#initialization)
    2. [Data Storage and Access](#datastorage)
    3. [Data I/O](#dataio)
    4. [Model Simulation](#modelsim)
        - [Calculating Tau Matrix](#calctau)
        - [Calculating P and Pi vectors](#calcppi)
        - [Calculating Additional Values](#calcother)
        - [Updating Core Model Parameters](#updatecore)
4. [Technical Implementation Details](#technical)

## Introduction <a name="introduction"></a>

This package contains a combination of python, C++, CUDA, and julia code to run the trade model simulation of the Long-run-growth project. The code is organized in such a way that the most performance sensitive and technical aspects of the codebase are handled in sufficient rigor by the lower level code while still allowing easy access to the overall model from a high-level, Python, object-based interface. This is done to combine the benefits of multiple languages and coding paradigms such that the code can be as performant as possible while still being accessible--at least at a high level--by somebody with minimal experience in low level and/or mathematical programming.

This documentation will cover, on one end, the high level interface to access the model and data through python, while also detailing the necessary steps to compile and run the entire codebase, and, finally, the more technical implementation details to allow for modification of any layer of the modular codebase.

## Installation <a name="installation"></a>

### Requirements <a name="requirements"></a>

- git
- Python 3.11 or greater
- A python virtual environment manager 
  - i.e. pyenv w/ pyenv-virtualenv or conda
- gcc compiler
- GNU Make
- For full GPU accelerated features:
  - A CUDA compatible GPU 
  - A CUDA installation including:
    - cublas v2
    - cudart 
  - nvcc compiler

Note that this installation procedure should be identical across most Unix systems (Linux/MacOS) however in order to properly install and run the package on Windows several changes would need to be made, in particular to the compilation and loading of the C++ shared libraries.

### Walkthrough <a name="walkthrough"></a>

1. Clone the repo into the desired location
2. Install the required python packages (ideally within a virtual environment):
  - numpy 
  - juliacall
3. From the project root directory run `make all` in the terminal to compile the C++/CUDA source code into shared libraries (.so files) in the lib folder
4. If not already present make a folder called `data` in project root directory, this is where input and output data will be placed for the model
5. To call the model from the associated python module either create a python script in the `src` directory and run it from the project root directory with `python src/<script_name.py>` or load model interface from a REPL or interactive shell started from the project root directory (in which case the core module should be imported with `from src.sim.core import *`)

## Usage <a name="usage"></a>

### Initialization <a name="initialization"></a>

Within a script, the core model implementation can be imported with
```python
from sim.core import *
```

Then, it is necessary to create a `Model` object with
```Python
<model_name> = Model(<model_size>)
```
Where `model_size` is either an integer variable or literal representing the number of active nodes within the model. 

### Data Storage and Access <a name="datastorage"></a>

Upon creation, the `Model` object will allocate the necessary memory to hold its data through the C++ API, these data arrays are exposed to the end Python user through the use of numpy arrays that are members of the respective Model object and can be accessed both for reference and assignment through 
```Python
<model_name>.<array_name>
```

The currently accessible arrays are:
1. `A` -- the vector of cell knowledge endowments
2. `L` -- the vector of cell labor (population) endowments
3. `B` -- the vector of cell inherent fertility values
4. `Y` -- the vector of cell total incomes
5. `P` -- the vector of cell ...
6. `Pi` -- the vector of cell ...
7. `t` -- the matrix (dim 2 array) of direct inter-cell travel costs
8. `tau` -- the matrix (dim 2 array) of globally aggregated inter-cell travel costs
9. `X` -- the matrix (dim 2 array) of ...
10. `Xi` -- the matrix (dim 2 array) of ...

Note that members 1-6 are of length `<model_size>` and 7-10 are technically of length `<model_size> * <model_size>` but in practical terms should be accessed as nested arrays, i.e. to get the direct travel cost between cells 100 and 101 call `<model_name>.t[100][101]`.


### Data I/O <a name="dataio"></a>

Once initialized, the `Model` member arrays can be accessed, copied, or modified at will the same way any standard numpy array would be. This allows for a high degree of flexibility to integrate this interface into 
any desired codebase or processing pipeline. However some simple data i/o is provided within the `sim.io` module.
```python
from sim.io import *
```

- Data can be loaded from a csv file with the function `read_array(filepath: str, array: numpy.ndarray, array_len: int, indexed: bool = False)`.

The variable `indexed` is an optional boolean stating whether the csv file is indexed or not. By default the function will assume an unindexed csv file with a single floating point number on each line. If the csv is indexed (i.e. in the form `1,5.2` for a single line) then pass `True` to the indexed variable in the function call.

**Example:**
`read_array('data/A.csv', my_model.A, 1861, True)`

- Data can be written to a csv file with the function `write_array(filepath: str, array: numpy.ndarray, array_len: int)`.

**Example:**
`write_array('data/P.csv', my_model.P, 1861)`

- Similarly, the `read_matrix(filepath: str, matrix: numpy.ndarray, array_len: int)` and `write_matrix(filepath: str, matrix: numpy.ndarray, array_len: int)` functions can be used to read or write matrices to and from a csv file of the form `<i>,<j>,<value>`

**Example:**
`read_matrix('data/t.csv', my_model.t, 1861)`

### Model Simulation <a name="modelsim"></a>

#### Calculating Tau matrix <a name="calctau"></a>

Once the relevant model parameters have been loaded, tau values can be calculated from the input t values through the class method 
```Python
<model_name>.calc_tau(coeff_theta: float, n_iterations: int, debug_level: int) 
```
A theta coefficient is required as well as the desired number of iterations of the tau approximation algorithm (for more details see the technical implementation section) and the debug level which should be an integer between 0 and 3 (default 0) representing the desired detail of debug output in the terminal.

Example:
```Python
model.calc_tau(3.0, 100, 1) 
```


#### Calculating P and Pi vectors <a name="calcppi"></a>

As the calculation of the optimal P and Pi vectors is quite sensitive a thorough initial solution should be found using a powerful non-linear optimization algorithm. This is currently implemented in julia and called from the python interface through 
```Python
<model_name>.init_p_pi(coeff_theta: str)
```
As this may take a significant amount of time to run, if multiple simulations of the model with the same starting conditions are expected it is recommended to save and load the results of this initial P/Pi calculation to speed up future runs of the model.

After the initial calculation, a much faster albeit less accurate iterative algorithm can be called to update the P and Pi vectors with 
```Python
<model_name>.calc_p_pi(coeff_theta: str, diff_limit: float, debug_level: int, relative_diff: bool = False) 
```
The `diff_limit` variable represents the desired level of iterative 'stability' for the calculation, i.e. the algorithm will iteratively solve for values of P and Pi until the largest difference in resulting values between iterations is smaller than this number. The `relative_diff` boolean flag (default `False`) indicates whether this `diff_limit` should be applied to the absolute or relative difference.

#### Calculating Additional Values <a name="calcother"></a>

Other calculation of model variables can be accessed as class methods including (along with their relevant parameters):
- `calc_Y(coeff_theta, debug_level)`
- `calc_X(coeff_theta, debug_level)`
- `calc_Xi(coeff_theta, debug_level)`

#### Updating Core Model Parameters <a name="updatecore"></a>

Additional functions are provided to allow for updating the core parameters of the model. They can be accessed as class methods:
- `update_A(coeff_eta, coeff_beta, coeff_sigma, debug_level, normalized, log_A, translate_A`
  - The `normalized` flag is an optional boolean value (default False). When True it sets all Xi values for adjacent nodes to be a value between 0 and 1 scaled by the sum of all Xi values.
  - The `log_A` flag is an optional boolean value (default True). When True it scales the increase in A to log levels (normalized by the A value of the source node). When False it simply grows A additively in linear terms based on the result of the growth equation. 
  - The `translate_A` flag is an optional boolean value (default True). When True it scales the effective A values used in calculating their influence on neighbors so that the minimum value is always zero. This means that nodes with very low (for technical reasons nodes must have nonzero A) have no influence on generating growth in A.
- `update_L(coeff_a, coeff_b, coeff_f, coeff_d, coeff_xi, coeff_lambda, debug_level)`
- `update_t(coeff_chi, debug_level)`

## Technical implementation details <a name="technical"></a>
                                                                                                          a
**TODO**                                                                                                                                                                                                                                                                                                                                                                                                                                  o
