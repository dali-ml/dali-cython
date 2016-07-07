from dali.runtime_config cimport default_preferred_device

cpdef Device default_device():
    global default_preferred_device
    return Device.wrapc(default_preferred_device)

cpdef void set_default_device(dev):
    global default_preferred_device

    cdef Device py_device
    py_device = ensure_device(dev)

    default_preferred_device = py_device.o

cpdef Device ensure_device(object dev):
    cdef Device ret
    if isinstance(dev, str):
        return Device(dev)
    elif isinstance(dev, Device):
        return dev
    elif dev is None:
        return default_device()
    else:
        raise ValueError("Expected device, got " + dev)
    return ret


cdef class Device:
    def __cinit__(Device self, str dev):
        cdef int gpu_num

        self.o = CDevice.device_of_doom()

        if dev == 'cpu':
            self.o = CDevice.cpu()
        elif dev.startswith('gpu'):
            suffix = dev.lstrip('gpu')
            if suffix == '':
                self.o = CDevice.gpu(0)
            else:
                if suffix.startswith("/"):
                    try:
                        gpu_num = int(suffix.lstrip('/'))
                        self.o = CDevice.gpu(gpu_num)
                    except ValueError:
                        pass

        if self.o.type() == DEVICE_T_ERROR:
            raise ValueError("Expected device, got " + dev)

    @staticmethod
    cdef Device wrapc(CDevice cdev):
        cdef Device d
        d = Device('cpu')
        d.o = cdev
        return d

    def description(Device self, bint real_gpu_name=True):
        return self.o.description(real_gpu_name)

    def __str__(Device self):
        if self.o.is_cpu():
            return 'cpu'
        elif self.o.is_gpu():
            return 'gpu/{}'.format(self.o.number())
        elif self.o.is_fake():
            return 'fake/{}'.format(self.o.number())
        else:
            return 'device_error'

    def __repr__(Device self):
        return str(self)
