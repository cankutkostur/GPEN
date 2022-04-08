import argparse

import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='weights/weights.pth')
    parser.add_argument('--name', type=str, default='weights_g_512.pth')

    args = parser.parse_args()

    ckpt = torch.load(args.path)

    torch.save(ckpt['g'], args.name)
    
