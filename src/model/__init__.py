# ==================================================
# [path configuration]

import sys
import os
path_thisfile = os.path.dirname(os.path.abspath(__file__))
path_root = os.path.normpath(os.path.join(path_thisfile, '..', '..'))
if not path_root in sys.path:
    sys.path.append(path_root)

# ==================================================

from importlib import import_module

# ==================================================

dict_modulename =   {
                   'DnCNN_S': 'DnCNN',
                   'DnCNN_B': 'DnCNN',
                   'CDnCNN_S': 'DnCNN',
                   'CDnCNN_B': 'DnCNN',
                   # DIDN
                   'DIDN_6': 'DIDN',
                   'DIDN_8': 'DIDN',
                   # CLtoN
                   'C2N_D': 'C2N',
                   'C2N_G': 'C2N',
                    }

def get_model_func(name_model):
    if name_model is None:
        return None
    else:
        module_model = import_module('src.model.{}'.format(dict_modulename[name_model]))
        return getattr(module_model, name_model)