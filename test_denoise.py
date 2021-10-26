import argparse, os


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

    # device setting
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test(args)

def test(args):
    


if __name__ == '__main__':
    main()
