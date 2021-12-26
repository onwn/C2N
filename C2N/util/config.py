import os

import yaml


class ConfigParser:
    def __init__(self, args):
        # load model configuration
        path_config = os.path.join('C2N', 'config')
        fname_config = f'{os.path.splitext(os.path.basename(args.config))[0]}.yml'
        with open(os.path.join(path_config, fname_config)) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # load argument
        for arg in args.__dict__:
            self.config[arg] = args.__dict__[arg]

        # string None handing
        self.convert_None(self.config)

    def __getitem__(self, name):
        return self.config[name]

    def convert_None(self, d):
        for key in d:
            if d[key] == 'None':
                d[key] = None
            if isinstance(d[key], dict):
                self.convert_None(d[key])
