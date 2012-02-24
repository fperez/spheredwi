__all__ = ['save']

import numpy as np
import os

data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../tmp')

def temp_storage(f):
    def first_check_temp_dir(*args, **kwargs):
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        return f(*args, **kwargs)

    return first_check_temp_dir

@temp_storage
def save(file, arr):
    """Proxy to NumPy's save command that stores to a temporary directory.

    """

    file = os.path.join(data_path, file)
    return np.save(file, arr)

@temp_storage
def load(file):
    """Proxy to NumPy's load command that reads from a temporary directory.

    """

    file = os.path.join(data_path, file)
    return np.load(file)
