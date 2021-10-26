# region - path configuration
import sys
import os
path_thisfile = os.path.dirname(os.path.abspath(__file__))
path_root = os.path.normpath(os.path.join(path_thisfile, '..', '..'))
if not path_root in sys.path:
    sys.path.append(path_root)
# endregion - path configuration

from src.util.datatype import *

import numpy as np
import yaml
import copy

# ==================================================

# Easydict behaves in so weird way, so made custom class.
class Hdict(dict):
    """Hierarchical dict which also contains special dtypes."""
    def __init__(self, value=None, dtype=None):
        super(__class__, self).__init__()
        self.value = value
        self.dtype = dtype

    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, obj):
        self[attr] = obj

    def __call__(self):
        # self.dtype(self.value) is only needed for modifying.
        '''
        if self.value is None:
            return None
        return self.dtype(self.value)
        '''
        # just in this way,
        return self.value

    def _recur_get_with_key(self, target_Hdict, key_list):
        key = key_list[0]
        if key in target_Hdict.keys():
            if len(key_list) == 1:
                return target_Hdict[key].value
            elif len(key_list) > 1:
                return self._recur_get_with_key(target_Hdict[key], key_list[1:])
        else:
            return 'undef'
    def get_with_key(self, key_hier):
        key_list = key_hier.split('.')
        return self._recur_get_with_key(self, key_list)

    def __copy__(self):
        clone = __class__()
        for key in list(self.keys()):
            clone[key] = copy.copy(self[key])
        return clone
    
    def __deepcopy__(self, memo):
        clone = __class__()
        for key in list(self.keys()):
            clone[key] = copy.deepcopy(self[key])
        return clone
    
    # =====
    # TODO: to solve the multiprocessing __getstate__ error: https://stackoverflow.com/questions/25382455/python-notimplementederror-pool-objects-cannot-be-passed-between-processes
    # still don't understand why multiprocessing from datahandler/denoiseDS needs these.......
    # '''
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if hasattr(self_dict, 'pool'):  # TODO: do I need these lines?
            del self_dict['pool']
        return self_dict
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    # '''

class Config(Hdict):
    def __init__(self):
        super(__class__, self).__init__()

        # path
        self._init_path()

        # general
        self._init_general()

    def _init_path(self):
        if 'path' not in list(self.keys()):
            self.path = Hdict()
        self.path.root = Hdict(value=path_root, dtype=str_flexible)
        self.path.src = Hdict(value=os.path.join(self.path.root(), 'src'), dtype=str_flexible)
        self.path.preset = Hdict(value=os.path.join(self.path.src(), 'preset'), dtype=str_flexible)
        self.path.output = Hdict(value=os.path.join(self.path.root(), 'output'), dtype=str_flexible)
        self.path.prep = Hdict(value=os.path.join(self.path.root(), 'prep'), dtype=str_flexible)
        self.path.custom = Hdict(value=os.path.join(self.path.root(), 'custom'), dtype=str_flexible)

        if 'dset' not in list(self.path.keys()):
            self.path.dset = Hdict()
        self.path.dset.root = Hdict(value=os.path.join(self.path.root(), 'dataset'), dtype=str_flexible)
        self.path.dset.BSD68 = Hdict(value=os.path.join(self.path.dset.root(), 'CBSD68'), dtype=str_flexible)
        self.path.dset.Set68 = Hdict(value=os.path.join(self.path.dset.root(), 'Set68'), dtype=str_flexible)
        self.path.dset.Set68_44 = Hdict(value=os.path.join(self.path.dset.root(), 'Set68_44'),
                                        dtype=str_flexible)
        self.path.dset.BSDtrain400 = Hdict(value=os.path.join(self.path.dset.root(), 'BSDtrain400'),
                                           dtype=str_flexible)
        self.path.dset.BSD432 = Hdict(value=os.path.join(self.path.dset.root(), 'BSD432'), dtype=str_flexible)
        self.path.dset.DIV2K = Hdict(value=os.path.join(self.path.dset.root(), 'DIV2K'), dtype=str_flexible)
        self.path.dset.Urban100 = Hdict(value=os.path.join(self.path.dset.root(), 'Urban100'),
                                        dtype=str_flexible)
        self.path.dset.SIDD = Hdict(value=os.path.join(self.path.dset.root(), 'SIDD'), dtype=str_flexible)
        self.path.dset.SIDDplus = Hdict(value=os.path.join(self.path.dset.root(), 'SIDD+'), dtype=str_flexible)
        # self.path.dset.LGEDD = Hdict(value=os.path.join(self.path.dset.root(), 'LGE-Denoise-Dataset'),
        #                              dtype=str_flexible)
        # self.path.dset.LGEDD_in = Hdict(value=os.path.join(self.path.dset.root(), 'LGEDD_in'),
        #                                 dtype=str_flexible)
        self.path.dset.DND = Hdict(value=os.path.join(self.path.dset.root(), 'DND'), dtype=str_flexible)
        self.path.dset.Set14 = Hdict(value=os.path.join(self.path.dset.root(), 'Set14'), dtype=str_flexible)
        self.path.dset.Set12 = Hdict(value=os.path.join(self.path.dset.root(), 'Set12'), dtype=str_flexible)
        self.path.dset.Set5 = Hdict(value=os.path.join(self.path.dset.root(), 'Set5'), dtype=str_flexible)
        self.path.dset.PolyU_cropped = Hdict(value=os.path.join(self.path.dset.root(), 'PolyU', 'CroppedImages'),
                                             dtype=str_flexible)
        self.path.dset.ccnoise_cropped = Hdict(value=os.path.join(self.path.dset.root(), 'ccnoise', 'cropped'),
                                               dtype=str_flexible)
    
    def _init_general(self):
        # session name
        self.session = Hdict(value=None, dtype=str_flexible)

        # device
        if 'device' not in list(self.keys()):
            self.device = Hdict()
        self.device.GPU_id = Hdict(value=None, dtype=str_flexible)    # if None, no GPU is used.
        self.device.n_thread = Hdict(value=None, dtype=int_flexible)    # if 0, load data in the main process.
        self.device.RAM_thres_GB = Hdict(value=None, dtype=float_flexible)
    
    # ==================================================
    # basic operations

    def print_summary(self, list_key):
        print('{:-^70}'.format('config summary'))
        for key_hier in list_key:
            print('{}: {}'.format(key_hier, self.get_with_key(key_hier)))
        print('{:-^70}'.format(''))
    
    def modify(self, nested_key, new_value):
        split_key = nested_key.split('.')
        self._recur_modify_with_key(self, split_key, new_value)

    def _recur_modify_with_dict(self, org_Hdict, new_dict):
        # the new_dict mustn't be a Hdict.
        # print(new_dict)
        for str_attr, new_value in new_dict.items():
            if str_attr in org_Hdict:
                if isinstance(new_value, dict):
                    self._recur_modify_with_dict(org_Hdict[str_attr], new_value)
                else:
                    if org_Hdict[str_attr].dtype is None:
                        print("preset attr is not a container of a value: {}".format(str_attr))
                    else:
                        org_Hdict[str_attr].value = org_Hdict[str_attr].dtype(new_value)
            else:
                # print(org_Hdict)
                # print("ignored preset attr: {}".format(str_attr))
                pass
    def modify_with_preset(self, name_preset):
        pathname_preset = os.path.join(self.path.preset(), name_preset+'.yml')
        with open(pathname_preset, 'r') as f:
            # self._recur_modify_with_dict(self, yaml.load(f))
            self._recur_modify_with_dict(self, yaml.load(f, Loader=yaml.FullLoader))

    def _recur_modify_with_key(self, org_Hdict, key_list, new_value):
        if len(key_list) == 1:
            key = key_list[0]
            org_Hdict[key].value =  org_Hdict[key].dtype(new_value)
        elif len(key_list) > 1:
            self._recur_modify_with_key(org_Hdict[key_list[0]], key_list[1:], new_value)
        else:
            print("invalid config attribute")
    def modify_with_argparse(self, parser):
        args = vars(parser.parse_args())
        if args['preset'] != 'undef':
            self.preset = Hdict(value=args['preset'], dtype=str_flexible)
            self.modify_with_preset(args['preset'])

        for key_arg, new_value in args.items():
            if new_value != 'undef':
                key_list = key_arg.split('.')
                self._recur_modify_with_key(self, key_list, new_value)
            else:
                pass
    def general_parser(self):
        import argparse

        parser = argparse.ArgumentParser()
        # preset
        parser.add_argument('--preset', '-p', dest='preset', type=str, default='undef')
        # below modifications will override the self from a preset file.

        # ==================================================
        # [general]

        # session name
        parser.add_argument('--session', '-s', dest='session',
                            type=str_flexible, default='undef')

        # device
        parser.add_argument('--device.GPU_id', '-g', dest='device.GPU_id',
                            type=str_flexible, default='undef')
        parser.add_argument('--device.n_thread', '-n_th',
                            dest='device.n_thread', type=int_flexible,
                            default='undef')
        parser.add_argument('--device.RAM_thres_GB', '-memory',
                            dest='device.RAM_thres_GB', type=float_flexible,
                            default='undef')

        return parser

    # ==================================================
    # setup operations

    def set_CUDA(self):
        if self.device.GPU_id() is None:
            self.device.n_GPU = Hdict(value=0, dtype=int)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.device.GPU_id()
            ids = self.device.GPU_id().split(',')
            self.device.n_GPU = Hdict(value=len(ids), dtype=int)

# ==================================================

if __name__ == '__main__':
    pass