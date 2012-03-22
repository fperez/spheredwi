__all__ = ['save']

import numpy as np
import os

data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../tmp')

def temp_storage(f):
    def first_check_temp_dir(file, *args, **kwargs):
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        args = (os.path.join(data_path, file),) + args[1:]
        return f(*args, **kwargs)

    first_check_temp_dir.__doc__ = f.__doc__
    return first_check_temp_dir

@temp_storage
def save(file, arr):
    """Proxy to NumPy's save command that stores to a temporary directory.

    """
    return np.save(file, arr)

@temp_storage
def savez(file, *args, **kwargs):
    """Proxy to NumPy's savez command that stores to a temporary directory.

    """
    return np.savez(file, *args, **kwargs)

@temp_storage
def load(file):
    """Proxy to NumPy's load command that reads from a temporary directory.

    """
    return np.load(file)

@temp_storage
def remove(file):
    """Proxy to NumPy's savez command that stores to a temporary directory.

    """
    import os
    os.unlink(file)
