import os
import torch
import numpy as np
from PIL import Image


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def atten2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor[0]
    image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy / (image_numpy.max() / 255.0)
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
