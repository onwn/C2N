# region - path configuration
import sys
import os
path_thisfile = os.path.dirname(os.path.abspath(__file__))
path_root = os.path.normpath(os.path.join(path_thisfile, '..', '..'))
if not path_root in sys.path:
    sys.path.append(path_root)
# endregion - path configuration

import numpy as np

# ==================================================

def str_flexible(v):
    if v == 'undef':
        return 'undef'
    elif (v is None) or (v == 'None'):
        return None
    elif isinstance(v, str):
        return v
    else:
        return str(v)
        # print("invalid string argument: {}".format(v))

def bool_flexible(v):
    if v == 'undef':
        return 'undef'
    elif (v is None) or (v == 'None'):
        return None
    elif isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        print("invalid boolean argument: {}".format(v))

def int_flexible(v):
    if v == 'undef':
        return 'undef'
    elif (v is None) or (v == 'None'):
        return None
    elif isinstance(v, int):
        return v
    else:
        return int(v)
        # print("invalid integer argument: {}".format(v))

def float_flexible(v):
    if v == 'undef':
        return 'undef'
    elif (v is None) or (v == 'None'):
        return None
    elif isinstance(v, float):
        return v
    elif isinstance(v, str) and '/' in v:
        numerator, denominator = v.split('/')
        return float(numerator) / float(denominator)
    else:
        return float(v)
        # print("invalid float argument: {}".format(v))

class ndarray_flexible():
    def __init__(self, dtype):
        self.dtype=dtype
    
    # TODO: dunno how to handle if the input is a string like '[1., 2., 3.]'
    def __call__(self, v):
        # TODO: how to compare the whole v with None or 'undef'?
        if isinstance(v, np.ndarray):
            return v.astype(self.dtype)
        elif isinstance(v, list) or isinstance(v, tuple):
            return np.array(v)
        else:
            print("invalid ndarray argument: {}".format(v))

if __name__ == '__main__':
    pass