import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as alb
from albumentations.pytorch import ToTensorV2


def store_dataset(img_dir):
    """
    list存放图片文件夹中的所有图片及图片路径
    :param img_dir:
    :return:
    """
    images = []
    all_path = []
    assert os.path.isdir(img_dir), '%s is not a valid directory' % img_dir

    for root, _, fnames in sorted(os.walk(img_dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            img = np.array(Image.open(path).convert('RGB'))
            images.append(img)
            all_path.append(path)

    return images, all_path


def __scale_width(image, target_width):
    """
    按照比例缩放图像
    :param image:
    :param kwargs:
    :return:
    """
    ow, oh = image.shape[1], image.shape[0]

    if ow > oh:
        if ow == target_width:
            return image
        w = target_width
        h = int(target_width * oh / ow)
    else:
        if oh == target_width:
            return image
        w = int(target_width * ow / oh)
        h = target_width
    scale_img = cv2.resize(image, (w, h), cv2.INTER_CUBIC)
    return scale_img


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize':
        transform_list.append(alb.Resize(opt.fine_size, opt.fine_size, cv2.INTER_AREA))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(alb.RandomCrop(opt.fine_size, opt.fine_size))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(alb.Lambda(image=lambda img, **kwargs: __scale_width(img, target_width=opt.scale_width)))
        transform_list.append(alb.RandomCrop(opt.fine_size, opt.fine_size))

    if opt.isTrain and opt.color_jitter:
        transform_list.append(alb.ColorJitter(brightness=0.2, p=0.5))

    transform_list += [alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                       ToTensorV2()]
    transform = alb.Compose(transform_list)
    return transform


class PairDataset(Dataset):
    def __init__(self, opt):
        super(PairDataset, self).__init__()

        self.opt = opt
        self.root_dir = opt.data_root
        self.dir_A = os.path.join(self.root_dir, 'dataA')
        self.dir_B = os.path.join(self.root_dir, 'dataB')

        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, idx):
        A_img = self.A_imgs[idx % self.A_size]
        B_img = self.B_imgs[idx % self.B_size]
        A_path = self.A_paths[idx % self.A_size]
        B_path = self.B_paths[idx % self.B_size]

        if self.opt.isTrain and not self.opt.no_flip:
            flip_transform = alb.Compose([alb.Flip(p=0.5)], additional_targets={"image0": "image"})
            ft_aug = flip_transform(image=A_img, image0=B_img)
            A_img = ft_aug['image']
            B_img = ft_aug['image0']

        AtoB = self.opt.which_direction == 'AtoB'

        pre_transform = alb.Compose([
            alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={"image0": "image"})
        transform = get_transform(self.opt)

        if AtoB:
            aug = pre_transform(image=A_img, image0=B_img)
            org_img = aug['image']
            target_img = aug['image0']

            augmentations = transform(image=A_img)
            A_img = augmentations["image"]
        else:
            aug = pre_transform(image=B_img, image0=A_img)
            org_img = aug['image']
            target_img = aug['image0']

            augmentations = transform(image=B_img)
            A_img = augmentations["image"]

        return {'A': A_img, 'input': org_img, 'target': target_img, 'A_paths': A_path, 'B_paths': B_path}

    @staticmethod
    def name():
        return 'PairDataset'


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from options import TrainOptions
    opt = TrainOptions().parse()
    opt.data_root = './LoL/train'
    opt.resize_or_crop = 'resize'
    # opt.color_jitter = True

    dataset = PairDataset(opt)

    for i, data in enumerate(dataset):
        imgA = data['input']
        imgB = data['target']
        imgA = imgA.numpy()
        imgB = imgB.numpy()
        img = np.concatenate([imgA, imgB], axis=2)
        img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0

        image = img.astype(np.uint8)
        image = Image.fromarray(image)

        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.title('Image{}'.format(i + 1))
        plt.show()
