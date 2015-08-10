import os
import random


def process_files(files, mapper, reducer):
    if files == str:
        files = [files]
    for file in files:
        for element in mapper(file):
            for res in reducer(element):
                yield res

def discover_files(root_path, extension=None):
    for path, dirs, files in os.walk(root_path):
        for file in files:
            if extension is None or file.endswith(extension):
                yield os.path.join(path, file)

class Mapper(object):
    FILTER      = 1
    TRANSFORMER = 2
    def __init__(self, map_f):
        self.map_f      = map_f
        self._transformations = []

    def __call__(self, *args, **kwargs):
        for element in self.map_f(*args, **kwargs):
            ignore = False
            for transform_f in self._transformations:
                element = transform_f(element)
                if element is None:
                    ignore = True
                    break
            if ignore:
                continue
            yield element

    def add_filter(self, filter_f):
        def wrapper(element):
            if filter_f(element):
                return element
            return None
        self.add_transform(wrapper)
        return self

    def add_transform(self, transform_f):
        self._transformations.append(transform_f)
        return self

class LineExtractor(Mapper):
    def __init__(self):
        def extract_lines(file):
            with open(file, "rt") as f:
                for line in f:
                    yield line[:-1]
        super(LineExtractor, self).__init__(extract_lines)

    def lower(self):
        return self.add_transform(lambda x: x.lower())


    def bound_length(self, lower_bound=None, upper_bound=None):
        if lower_bound:
            self.add_filter(lambda x: lower_bound <= len(x))
        if lower_bound:
            self.add_filter(lambda x: len(x) <= upper_bound)
        return self

    def split_spaces(self):
        return self.add_transform(lambda x: x.split(' '))

def batched_reducer(minibatch_size,
                    minibatch_f=lambda x:x,
                    examples_until_minibatches=None,
                    sorting_key=lambda x: len(x)):
    collected = []
    examples_until_minibatches = examples_until_minibatches or minibatch_size
    def wrapper(el):
        collected.append(el)
        if len(collected) >= examples_until_minibatches:
            collected.sort(key=sorting_key)
            batch_start_idxes = list(range(0, len(collected), minibatch_size))
            random.shuffle(batch_start_idxes)
            for i in batch_start_idxes:
                if i + minibatch_size < len(collected):
                    yield minibatch_f(collected[i:(i + minibatch_size)])
    return wrapper
