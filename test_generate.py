import argparse, os

import numpy as np
import cv2
import torch

from src.util.config import ConfigParser
from src.model import get_model_func

def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('--config',       default=None,     type=str)
    args.add_argument('--ckpt',         default=None,     type=str)
    args.add_argument('--mode',         default='single', type=str)
    args.add_argument('--data',         default=None,     type=str)
    args.add_argument('--gpu',          default=None,     type=int)

    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'
    assert args.ckpt is not None, 'checkpoint epoch is needed'
    assert args.data is not None, 'data path or filename is needed'
    assert args.mode in ['single', 'dataset'], 'mode must be single or dataset'

    # device setting
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    configs = ConfigParser(args)

    denoise(configs)

def denoise(configs):
    # model load
    generator = get_model_func(configs['model']['Generator'])()
    if configs['gpu'] is not None: generator = generator.cuda()
    ckpt = torch.load(configs['ckpt'])
    generator.load_state_dict(ckpt)
    print('model loaded!')

    # make results folder
    os.makedirs('./results', exist_ok=True)

    # denoise
    if configs['mode'] == 'single':
        generated = generate_single_img(configs, generator, configs['data'])
        cv2.imwrite('./results/%s'%(configs['data'].split('/')[-1][:-4] + '_generated.png'), generated)
        print('generated %s'%(configs['data']))
    elif configs['mode'] == 'dataset':
        for (dirpath, _, filenames) in os.walk(configs['data']):
            folder_name = dirpath.split('/')[-1] if dirpath.split('/')[-1] != '' else dirpath.split('/')[-2]
            os.makedirs('./results/%s'%folder_name, exist_ok=True)

            for filename in filenames:
                generated = generate_single_img(configs, generator, os.path.join(dirpath, filename))
                cv2.imwrite('./results/%s/%s'%(folder_name, filename[:-4] + '_generated.png'), generated)
                print('generated %s'%(configs['data']))

def generate_single_img(configs, generator, img_path):
    img = cv2.imread(img_path, -1).astype(float)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)
    if configs['gpu'] is not None: img = img.cuda()

    generated = generator(img)
    generated = generated.cpu().detach().squeeze(0).numpy()
    generated = generated.transpose(1, 2, 0)
    generated = generated * 255.0
    generated = generated.astype(np.uint8)

    return generated

if __name__ == '__main__':
    main()
