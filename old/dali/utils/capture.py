from collections import defaultdict

class Capture(object):
    instances = set()

    def __init__(self):
        self.state = defaultdict(lambda: [])

    @classmethod
    def add(cls, name, value):
        for instance in cls.instances:
            instance.state[name].append(value)

    def __enter__(self):
        Capture.instances.add(self)

    def __exit__(self, *args, **kwargs):
        Capture.instances.remove(self)
