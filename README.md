# Dali Performance Tests with Python Bindings

Test for the perfomance of bindings using Cython of [Dali](https://github.com/JonathanRaiman/Dali). All templated types are turned into `double`, and very little functionality is carried over.

@author Jonathan Raiman
@date 18th April 2015

### Installation

```bash
python3 setup.py install
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
