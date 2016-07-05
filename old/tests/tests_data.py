import dill as pickle
import mock
import unittest

from dali.data import DiscoverFiles, Lines, BatchBenefactor

from os.path import join, dirname, realpath
SCRIPT_DIR = dirname(realpath(__file__))

class DataTests(unittest.TestCase):
    @mock.patch('os.walk')
    def test_discover_files(self, patched_os_walk):
        patched_os_walk.return_value = [
            ('test_dir', ['hi'], ['lol.py', 'lol2.py']),
            ('test_dir/hi', [], ['hello.py', 'yikes.txt']),
        ]

        x = DiscoverFiles('/what/ever', '.py')

        self.assertEqual(next(x), "test_dir/lol.py")
        x_pickle = pickle.loads(pickle.dumps(x))

        self.assertEqual(next(x), "test_dir/lol2.py")
        self.assertEqual(next(x_pickle), "test_dir/lol2.py")

        self.assertEqual(next(x), "test_dir/hi/hello.py")
        self.assertEqual(next(x_pickle), "test_dir/hi/hello.py")

        with self.assertRaises(StopIteration):
            next(x)
        with self.assertRaises(StopIteration):
            next(x_pickle)

    def test_lines(self):
        r = Lines() \
            .lower()                                 \
            .split_spaces()                          \
            .bound_length(2,4)

        r.set_file(join(SCRIPT_DIR, "test.txt"))

        self.assertEqual(next(r), ["ala", "ma", "kota"])

        r2 = pickle.loads(pickle.dumps(r))

        self.assertEqual(next(r), ["gdzie", "jest","ala", "?"])
        self.assertEqual(next(r2), ["gdzie", "jest","ala", "?"])

        with self.assertRaises(StopIteration):
            next(r)
        with self.assertRaises(StopIteration):
            next(r2)


    @mock.patch('random.shuffle')
    def test_batch_benefactor(self, patched_random_shuffle):
        # random shuffle no longer shuffles ;-)

        d = BatchBenefactor(2, lambda x: ' '.join(x), 4)
        d.add("where")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("is")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("the")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("love")
        self.assertEqual(list(d), ["is the", "love where"])

        d = pickle.loads(pickle.dumps(d))
        d.add("where")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("is")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("the")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("love")
        self.assertEqual(list(d), ["is the", "love where"])

        d = pickle.loads(pickle.dumps(d))
        d.update_minibatch_size(4)
        d.add("siema")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("is")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("polish")
        with self.assertRaises(StopIteration):
            next(d)
        d.add("greeting")
        self.assertEqual(list(d), ["is siema polish greeting"])
        with self.assertRaises(StopIteration):
            next(d)
