from dali.runtime_config cimport default_preferred_device
include "../../config.pxi"

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
        raise ValueError("Expected device, got " + str(dev))
    return ret


cdef class Device:
    def __cinit__(Device self, str dev):
        cdef int gpu_num

        self.o = CDevice.device_of_doom()

        if dev == 'cpu':
            self.o = CDevice.cpu()
            return
        IF DALI_USE_CUDA:
            if dev.startswith('gpu'):
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
                return
        ELSE:
            if dev.startswith('gpu'):
                raise ValueError(
                    "Dali compiled without CUDA support cannot "
                    "construct gpu devices (got device='" + dev +
                    "')."
                )
        if self.o.type() == DEVICE_T_ERROR:
            raise ValueError("Expected device, got " + dev)

    @staticmethod
    cdef Device wrapc(CDevice cdev):
        cdef Device d
        d = Device('cpu')
        d.o = cdev
        return d

    def name(Device self):
        """
        Returns the pretty name of the device.
        If the device is a GPU, returns
        the true name and model of the device
        """
        return self.o.description(True).decode("utf-8")

    IF DALI_USE_CUDA:
        def gpu_name(Device self):
            return self.o.gpu_name().decode("utf-8")

    def __str__(Device self):
        if self.o.is_cpu():
            return "Device('cpu')"
        IF DALI_USE_CUDA:
            if self.o.is_gpu():
                return "Device('gpu/{}')".format(self.o.number())
        if self.o.is_fake():
            return "Device('fake/{}')".format(self.o.number())
        return 'Device(???)'

    def __repr__(Device self):
        return str(self)
