# Dali in Python

This repository contains an early version of Python bindings to [Dali](https://github.com/JonathanRaiman/Dali), the automatic differentiation framework.

With this module you can construct arbitrary mathematical models, including differentiating through control code and loops, optimize and run them using your GPU or CPU.

The bindings are light-weight, and the resulting module retains about 90%-95% of the C++ performance in most use-cases (with this performance going up for larger models and GPU intensive computations).

@author Jonathan Raiman & Szymon Sidor
@date 20th March 2016

### Installation

1. Install [Dali](https://github.com/JonathanRaiman/Dali#installation) using `homebrew`, `yum`, or `apt-get`.

2. Install the Python package:

```bash
pip3 install -r requirements.txt
python3 setup.py install
```

### Usage

See examples under notebooks.
