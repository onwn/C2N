import argparse
import os

import cv2
import numpy as np
import torch

from src.model import get_model
from src.util.config import ConfigParser


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('--config', default=None, type=str)
    args.add_argument('--ckpt', default=None, type=str)
    args.add_argument('--mode', default='single', type=str)
    args.add_argument('--data', default=None, type=str)
    args.add_argument('--gpu', default=None, type=int)

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

@torch.no_grad()
def denoise(configs):
    # model load
    denoiser = get_model(configs['model']['Denoiser'])
    if configs['gpu'] is not None:
        denoiser = denoiser.cuda()
    ckpt = torch.load(configs['ckpt'])
    denoiser.load_state_dict(ckpt)
    denoiser.eval()
    print('model loaded!')

    # make results folder
    os.makedirs('./results', exist_ok=True)

    # denoise
    if configs['mode'] == 'single':
        denoised = denoise_single_img(configs, denoiser, configs['data'])
        fname_data = os.path.basename(configs['data'])
        tag_data = os.path.splitext(fname_data)[0]
        fpath_output = f'./results/{tag_data}_denoised.png'
        cv2.imwrite(fpath_output, denoised)
        print('denoised to %s' % (fpath_output))
    elif configs['mode'] == 'dataset':
        for (dirpath, _, filenames) in os.walk(configs['data']):
            folder_name = os.path.basename(os.path.normpath(dirpath))
            os.makedirs('./results/%s' % folder_name, exist_ok=True)

            for filename in filenames:
                denoised = denoise_single_img(configs, denoiser,
                                              os.path.join(dirpath, filename))
                tag_data = os.path.splitext(filename)[0]
                fpath_output = f'./results/{folder_name}/{tag_data}_denoised.png'
                cv2.imwrite(fpath_output, denoised)
                print('denoised to %s' % (fpath_output))


def denoise_single_img(configs, denoiser, img_path):
    img = cv2.imread(img_path, -1).astype(float)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)
    if configs['gpu'] is not None:
        img = img.cuda()

    denoised = denoiser(img)
    denoised = denoised.cpu().detach().squeeze(0).numpy()
    denoised = denoised.transpose(1, 2, 0)
    denoised = denoised * 255.0
    denoised += 0.5
    denoised = np.clip(denoised, 0., 255.)
    denoised = denoised.astype(np.uint8)

    return denoised


if __name__ == '__main__':
    main()
