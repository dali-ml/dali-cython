import os
import random

from queue import Queue

from dali.data.utils import split_punctuation as split_punctuation_f

class Process(object):
    def __init__(self, files, mapper, reducer):
        if files == str:
            files = (make_me_iterator for make_me_iterator in [files])
        self.files   = files
        self.mapper  = mapper
        self.reducer = reducer

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return next(self.reducer)
            except StopIteration:
                pass
            # if we got here it means reducer run out of elements
            # and we need to add more from mapper.
            try:
                self.reducer.add(next(self.mapper))
                continue
            except StopIteration:
                pass
            # if we got here it means that mapper run out of elements and we need to give
            # if another file
            next_file = next(self.files)
            self.mapper.set_file(next_file)

class DiscoverFiles(object):
    def __init__(self, root_path, extension=None):
        self.files = []
        self.next_file = 0
        for path, dirs, files in os.walk(root_path):
            for file in files:
                if extension is None or file.endswith(extension):
                    self.files.append(os.path.join(path, file))

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_file >= len(self.files):
            raise StopIteration()
        else:
            self.next_file += 1
            return self.files[self.next_file - 1]


class FileMapper(object):
    FILTER      = 1
    TRANSFORMER = 2

    def __init__(self):
        self._transformations = []
        self.file_name = None
        self.file_handle = None

    def set_file(self, file_name):
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
        self.file_name = file_name
        self.file_handle = None


    def get_file(self, fargs="rt"):
        if self.file_handle is None:
            self.file_handle = open(self.file_name, fargs)
        return self.file_handle

    def __del__(self):
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

    def __next__(self):
        if self.file_name is None:
            raise StopIteration()
        while True:
            item = self.next_item_no_filter()
            if item is not None:
                return item

    def next_item_no_filter(self):
        """By default __next__ does not even return an item
           if it does not pass a filer. This function, returns
           None even if the filters are not passing. This
           is useful for example for translation where,
           we want to ensure that two different file streams
           are always aligned
        """
        if self.file_name is None:
            raise StopIteration()

        item = self.next_item()
        item = self.transform_item(item)

        return item

    def transform_item(self, item):
        for transform_f in self._transformations:
            item = transform_f(item)
            if item is None:
                break
        return item

    def next_item(self):
        raise StopIteration()


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

    def __iter__(self):
        return self

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['file_handle']
        if self.file_handle is not None:
            state['__file_position'] = self.file_handle.tell()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        self.file_handle = open(self.file_name)
        if '__file_position' in state:
            self.file_handle.seek(state['__file_position'])


class Lines(FileMapper):
    def next_item(self):
        res = self.get_file().readline()
        if len(res) == 0:
            raise StopIteration()
        if len(res) > 0 and res[-1] == '\n':
            return res[:-1]
        else:
            return res

    def lower(self):
        return self.add_transform(lambda x: x.lower())

    def bound_length(self, lower_bound=None, upper_bound=None):
        if lower_bound:
            self.add_filter(lambda x: lower_bound <= len(x))
        if upper_bound:
            self.add_filter(lambda x: len(x) <= upper_bound)
        return self

    def split_spaces(self):
        return self.add_transform(lambda x: x.split(' '))

    def reverse(self):
        return self.add_transform(lambda x: list(reversed(x)))

    def split_punctuation(self):
        return self.add_transform(split_punctuation_f)


class Multiplexer(object):
    def __init__(self, *mappers):
        self.mappers = mappers

    def set_file(self, args):
        assert len(args) == len(self.mappers)

        for mapper, arg in zip(self.mappers, args):
            mapper.set_file(arg)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            ret = tuple([m.next_item_no_filter() for m in self.mappers])
            if all(ret_i is not None for ret_i in ret):
                return ret



class BatchBenefactor(object):
    def __init__(self, minibatch_size,
                 minibatch_class=None,
                 examples_until_minibatches=None,
                 sorting_key=None):
        self.minibatch_size = minibatch_size
        self.minibatch_class = minibatch_class
        self.examples_until_minibatches = examples_until_minibatches or minibatch_size
        self.sorting_key = sorting_key

        self.collected = []
        self.batches = []
        self.next_batch = 0

    def add(self, element):
        self.collected.append(element)

        if len(self.collected) >= self.examples_until_minibatches:
            self.batches = []
            self.next_batch = 0

            sorting_key = self.sorting_key or (lambda x: len(x))
            self.collected.sort(key=sorting_key)


            batch_start_idxes = list(range(0, len(self.collected), self.minibatch_size))
            random.shuffle(batch_start_idxes)
            for i in batch_start_idxes:
                if i + self.minibatch_size <= len(self.collected):
                    self.batches.append(self.collected[i:(i + self.minibatch_size)])
            self.collected = []

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.batches) == 0 or self.next_batch == len(self.batches):
            self.batches = []
            self.next_batch = 0
            raise StopIteration()
        else:
            assert self.next_batch < len(self.batches)
            self.next_batch += 1
            minibatch_class = self.minibatch_class or (lambda x:x)
            return minibatch_class(self.batches[self.next_batch - 1])



    def update_minibatch_size(self, minibatch_size):
        self.minibatch_size = minibatch_size


class IdentityReducer(object):
    def __init__(self):
        self.q = Queue()

    def add(self, element):
        self.q.put(element)

    def __iter__(self):
        return self

    def __next__(self):
        if self.q.empty():
            raise StopIteration()
        else:
            return self.q.get()
