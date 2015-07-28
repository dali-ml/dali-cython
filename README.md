# Dali Performance Tests with Python Bindings

Test for the perfomance of bindings using Cython of [Dali](https://github.com/JonathanRaiman/Dali). All templated types are turned into `double`, and very little functionality is carried over.

@author Jonathan Raiman
@date 18th April 2015

### Installation

Add the following line to your .bashrc on Linux/GNU or .bash_profile on Mac OS:
```bash
export DALI_HOME="/path/to/dali/cpp/implementation"
```
Then you can simply install Dali by running:

```bash
python3 setup.py install
```

#### Special installation instructions for Linux

Since on Linux/GNU Dali needs to be compiled using shared libraries, we need to tell the OS where to load them from. Add the following line to your .bashrc:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DALI_HOME/build_cpu/dali
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DALI_HOME/build/dali
```

### Usage

```python

from test_dali import Mat, RNN

# perform simple operations on
# matrices
A = Mat(3, 3)
B = Mat(3, 3)
C = A + B

# pass A and B as input to RNN
rnn = RNN(3, 3)
D = rnn.activate(A, B)

```

For the optimizers, they are stored as:

```python

from test_dali import SGD

sgd = SGD()

params = [A, B]
step_size = 0.01
sgd.step(params, step_size)

```

