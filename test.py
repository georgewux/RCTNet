import cv2 as cv
import albumentations as alb
import random
import matplotlib.pyplot as plt
import numpy as np


# def visualize(image):
#     plt.figure(figsize=(6, 6))
#     plt.axis('off')
#     plt.imshow(image)
#     plt.show()
#
#
# def hflip_image(image, **kwargs):
#     ow, oh = image.shape[0], image.shape[1]
#     target_width = kwargs['target_width']
#
#     if ow > oh:
#         if ow == target_width:
#             return image
#         w = target_width
#         h = int(target_width * oh / ow)
#     else:
#         if oh == target_width:
#             return image
#         w = int(target_width * ow / oh)
#         h = target_width
#     img = cv.resize(image, (w, h), cv.INTER_CUBIC)
#     return img
#
#
# image = cv.imread(r'E:\WuWeiran\RCTNet\datasets\LoL\train\dataB\2.png')
# image0 = cv.imread(r'E:\WuWeiran\RCTNet\datasets\LoL\train\dataB\5.png')
# hflip = alb.Lambda(image=lambda img, p=0.5, **kwargs: hflip_image(img, target_width=512))
# transform = alb.Compose([hflip], additional_targets={"image0": "image"})
#
# random.seed(7)
# transformed = transform(image=image, image0=image0)
# img = np.concatenate([transformed['image'], transformed['image0']], axis=1)
# visualize(img)

import torch

x = torch.tensor([16])
# print(x.shape)
# y = torch.randn(5, 16, 64, 1)
#
#
# x = x.reshape(x.size(0), x.size(1), -1, 1)

# z = x @ y
# print(z.shape)

y = torch.sqrt(x)
print(y)

