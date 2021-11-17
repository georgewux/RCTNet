import os
import torch
from torch import nn
from models.networks import Vgg16


def load_vgg16(model_dir, device):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    vgg = Vgg16()
    vgg.to(device)
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))

    return vgg


def vgg_preprocess(batch, opt):  # 因为使用的是别人的预训练权重，所以模型输入的格式也必须和训练时保持一致
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    if opt.vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939  # 为每一个batch的该通道里的所有像素值赋予同一个值
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(mean)  # subtract mean
    return batch


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        if self.opt.no_vgg_instance:
            return torch.mean((img_fea - target_fea) ** 2)
        else:
            return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


if __name__ == '__main__':
    device = torch.device('cuda')

    x = torch.randn(1, 3, 512, 512).cuda()
    y = torch.randn(1, 3, 512, 512).cuda()

    from options import TrainOptions
    opt = TrainOptions().parse()
    opt.no_vgg_instance = True
    criterion = PerceptualLoss(opt)

    vgg = load_vgg16('./', device)

    dis = criterion.compute_vgg_loss(vgg, x, y)

    print(dis)
