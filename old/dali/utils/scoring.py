import numpy as np
import os
import subprocess
import stat
import tempfile

from os.path import join, exists
from urllib.request import urlretrieve

from dali.core import Mat
from .misc import subsample, median_smoothing

MULTIBLEU_SCRIPT = 'multi-bleu.perl'
MULTIBLEU_URL = "https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl"

class ErrorTracker(object):
    def __init__(self):
        self.epoch_error = []
        self.error_evolution = []

    def append(self, error):
        if type(error) == Mat:
            self.epoch_error.append(error.w[0,0])
        elif type(error) == np.ndarray:
            self.epoch_error.append(error[0,0])
        else:
            self.epoch_error.append(error)

    def finalize_epoch(self):
        if len(self.epoch_error) > 0:
            self.epoch_error = subsample(self.epoch_error, maximum_length=1000)
            self.error_evolution.append(self.epoch_error)
            self.epoch_error = []

    def raw(self):
        x = []
        y = []
        for epoch_no, error_epoch in enumerate(self.error_evolution):
            x.extend(float(epoch_no) + float(t) / len(error_epoch) for t in range(len(error_epoch)))
            y.extend(error_epoch)

        if len(y) > 100:
            y = median_smoothing(y, 30)
            x = x[:len(y)]
        return x,y

    def num_epochs(self):
        return len(self.error_evolution)

    def recent(self, tsteps=1):
        if len(self.epoch_error) == 0:
            return np.nan
        else:
            recent = self.epoch_error[-tsteps:]
            return sum(recent)/len(recent)

def bleu(reference, hypotheses, script_location=None):
    if script_location is None:
        script_location = join(tempfile.gettempdir(), MULTIBLEU_SCRIPT)
        if not exists(script_location):
            urlretrieve(MULTIBLEU_URL, script_location)
    else:
        assert(exists(script_location))


    def process_input(val):
        if type(val) == str:
            assert exists(val)
            return open(val)
        elif type(val) == list:
            ret = tempfile.NamedTemporaryFile("wt")
            for example in val:
                if len(example) > 0 and example[-1] == '\n':
                    ret.write(example)
                else:
                    ret.write(example + '\n')
            ret.seek(0)
            return ret

    try:
        reference  = process_input(reference)
        hypotheses = process_input(hypotheses)
        prefix = "BLEU = "

        try:
            res_str = subprocess.check_output([script_location, reference.name], stdin=hypotheses, universal_newlines=True)
        except PermissionError:
            current_permissions = os.stat(script_location).st_mode
            os.chmod(script_location, current_permissions | stat.S_IEXEC)
            res_str = subprocess.check_output([script_location, reference.name], stdin=hypotheses, universal_newlines=True)

        assert(res_str.startswith(prefix))
        res_str = res_str[len(prefix):]
        res_str = res_str.split(',')[0]
        return float(res_str)
    finally:
        if hasattr(reference, 'close'):
            reference.close()
        if hasattr(hypotheses, 'close'):
            hypotheses.close()

__all__ = [
    "ErrorTracker",
    "bleu"
]




