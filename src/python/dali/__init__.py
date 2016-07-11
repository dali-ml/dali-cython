from numpy import float32, float64, int32

from . import tensor
from .op import *

from .array.array import Array
from .tensor.tensor import Tensor

from .array.memory.device import (
    set_default_device,
    default_device,
    Device
)

from .tensor.tape import backward
