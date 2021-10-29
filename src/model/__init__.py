from importlib import import_module

dict_modulename = {
    'DnCNN_S': 'DnCNN',
    'DnCNN_B': 'DnCNN',
    'CDnCNN_S': 'DnCNN',
    'CDnCNN_B': 'DnCNN',
    # DIDN
    'DIDN_6': 'DIDN',
    'DIDN_8': 'DIDN',
    # C2N
    'C2N_D': 'C2N',
    'C2N_G': 'C2N',
}


def get_model(name_model):
    if name_model is None:
        return None
    else:
        module_model = import_module('src.model.{}'.format(dict_modulename[name_model]))
        model_class = getattr(module_model, name_model)
        return model_class()
