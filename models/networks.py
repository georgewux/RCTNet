import os
import torch
from torch import nn
import torch.nn.functional as F


#################################################
# Functions
#################################################
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def pad_tensor(input, divide):
    height_org, width_org = input.shape[2], input.shape[3]

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def define_encoder(opt, device):
    model = RCTEncoder(opt).to(device)
    model.apply(weights_init)
    return model


def define_bifpn(opt, device):
    model = BiFPNBlock(opt).to(device)
    model.apply(weights_init)
    return model


def define_G_rct(opt, device):
    model = GlobalRCT(opt).to(device)
    model.apply(weights_init)
    return model


def define_L_rct(opt, device):
    model = LocalRCT(opt).to(device)
    model.apply(weights_init)
    return model


#################################################
# Classes
#################################################
class RCTConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, ksize=3, stride=2, pad=1):
        super(RCTConvBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),
            nn.BatchNorm2d(output_nc),
            nn.SiLU(inplace=True)  # Swish activation
        )

    def forward(self, x):
        return self.model(x)


class BiFPNBlock(nn.Module):
    def __init__(self, opt):
        super(BiFPNBlock, self).__init__()

        self.epsilon = 0.0001  # 防止融合权重归一化时除0

        self.p4_td = RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1)
        self.p5_td = RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1)
        self.p6_td = RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1)

        self.p4_out = RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1)
        self.p5_out = RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1)
        self.p6_out = RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1)
        self.p7_out = RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1)

        self.down5 = RCTConvBlock(opt.fusion_filter, opt.fusion_filter)
        self.down6 = RCTConvBlock(opt.fusion_filter, opt.fusion_filter)
        down_list = [
            RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1)
        ]
        self.down7 = nn.Sequential(*down_list)

        # Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 3))
        self.w2 = nn.Parameter(torch.Tensor(3, 3))
        self.relu = nn.ReLU()

    def forward(self, inputs):
        p4_x, p5_x, p6_x, p7_x = inputs

        w1 = self.relu(self.w1)  # 保证w为非负
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon

        # Calculate Top-Down Pathway
        p7_td = p7_x
        # print(p6_x.shape)
        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, scale_factor=8))
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, scale_factor=2))

        # Calculate Bottom-Up Pathway
        p4_out = p4_td
        p5_out = self.p5_out(w2[0, 0] * p5_x + w2[1, 0] * p5_td + w2[2, 0] * self.down5(p4_out))
        p6_out = self.p6_out(w2[0, 1] * p6_x + w2[1, 1] * p6_td + w2[2, 1] * self.down6(p5_out))
        p7_out = self.p7_out(w2[0, 2] * p7_x + w2[1, 2] * p7_td + w2[2, 2] * self.down7(p6_out))

        return [p4_out, p5_out, p6_out, p7_out]


class RCTEncoder(nn.Module):
    def __init__(self, opt):
        super(RCTEncoder, self).__init__()

        self.init = nn.Sequential(
            RCTConvBlock(3, opt.num_filter),
            RCTConvBlock(opt.num_filter, opt.num_filter * 2),
            RCTConvBlock(opt.num_filter * 2, opt.num_filter * 4)
        )

        self.bottom = RCTConvBlock(opt.num_filter * 4, opt.num_filter * 8)
        self.middle = RCTConvBlock(opt.num_filter * 8, opt.num_filter * 16)
        top_list = [
            RCTConvBlock(opt.num_filter * 16, 1024, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1)
        ]
        self.top = nn.Sequential(*top_list)

        self.p4 = RCTConvBlock(opt.num_filter * 4, opt.fusion_filter, 3, 1, 1)
        self.p5 = RCTConvBlock(opt.num_filter * 8, opt.fusion_filter, 3, 1, 1)
        self.p6 = RCTConvBlock(opt.num_filter * 16, opt.fusion_filter, 3, 1, 1)
        self.p7 = RCTConvBlock(1024, opt.fusion_filter, 1, 1, 0)

    def forward(self, x):
        c4 = self.init(x)
        p4_x = self.p4(c4)
        c5 = self.bottom(c4)
        p5_x = self.p5(c5)
        c6 = self.middle(c5)
        p6_x = self.p6(c6)
        c7 = self.top(c6)
        p7_x = self.p7(c7)

        return [p4_x, p5_x, p6_x, p7_x]


class GlobalRCT(nn.Module):
    def __init__(self, opt):
        super(GlobalRCT, self).__init__()

        self.r_conv = nn.Sequential(
            RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 1, 1, 0),
            nn.Conv2d(opt.fusion_filter, opt.represent_feature * opt.ngf, kernel_size=(1, 1))
        )
        self.t_conv = nn.Sequential(
            RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 1, 1, 0),
            nn.Conv2d(opt.fusion_filter, 3 * opt.ngf, kernel_size=(1, 1))
        )
        self.act = nn.Softmax(dim=2)

    def forward(self, feature, p_high):
        h, w = feature.shape[2], feature.shape[3]
        f_r = feature.reshape(feature.size(0), -1, opt.represent_feature)
        r_g = self.r_conv(p_high)
        r_g = r_g.reshape(r_g.size(0), opt.represent_feature, opt.ngf)
        t_g = self.t_conv(p_high)
        t_g = t_g.reshape(t_g.size(0), 3, opt.ngf)

        attention = torch.bmm(f_r, r_g) / torch.sqrt(torch.tensor(opt.represent_feature))
        attention = self.act(attention)
        Y_G = torch.bmm(attention, t_g.transpose(1, 2))
        Y_G = Y_G.transpose(1, 2)
        return Y_G.reshape(Y_G.size(0), 3, h, w)


class LocalRCT(nn.Module):
    def __init__(self, opt):
        super(LocalRCT, self).__init__()

        self.opt =opt

        self.r_conv = nn.Sequential(
            RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1),
            nn.Conv2d(opt.fusion_filter, opt.represent_feature * opt.nlf, kernel_size=(3, 3), padding=(1, 1))
        )
        self.t_conv = nn.Sequential(
            RCTConvBlock(opt.fusion_filter, opt.fusion_filter, 3, 1, 1),
            nn.Conv2d(opt.fusion_filter, 3 * opt.nlf, kernel_size=(3, 3), padding=(1, 1))
        )
        self.act = nn.Softmax(dim=2)

    def forward(self, feature, p_low):
        nfeature, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(feature, self.opt.mesh_size)
        mesh_h = int(nfeature.shape[2] / 31)
        mesh_w = int(nfeature.shape[3] / 31)

        r_l = self.r_conv(p_low)
        t_l = self.t_conv(p_low)
        Y_L = torch.zeros(nfeature.size(0), 3, nfeature.size(2), nfeature.size(3))

        # Grid-wise
        for i in range(self.opt.mesh_size):
            for j in range(self.opt.mesh_size):
                # cp means corner points of a grid
                r_k = r_l[:, :, i, j].reshape(-1, self.opt.represent_feature, self.opt.nlf)
                cp = r_l[:, :, i, j + 1].reshape(-1, self.opt.represent_feature, self.opt.nlf)
                r_k = torch.cat((r_k, cp), dim=2)
                cp = r_l[:, :, i + 1, j].reshape(-1, self.opt.represent_feature, self.opt.nlf)
                r_k = torch.cat((r_k, cp), dim=2)
                cp = r_l[:, :, i + 1, j + 1].reshape(-1, self.opt.represent_feature, self.opt.nlf)
                r_k = torch.cat((r_k, cp), dim=2)

                t_k = t_l[:, :, i, j].reshape(-1, 3, self.opt.nlf)
                cp = t_l[:, :, i, j + 1].reshape(-1, 3, self.opt.nlf)
                t_k = torch.cat((t_k, cp), dim=2)
                cp = t_l[:, :, i + 1, j].reshape(-1, 3, self.opt.nlf)
                t_k = torch.cat((t_k, cp), dim=2)
                cp = t_l[:, :, i + 1, j + 1].reshape(-1, 3, self.opt.nlf)
                t_k = torch.cat((t_k, cp), dim=2)

                f_k = nfeature[:, :, i*mesh_h:(i+1)*mesh_h, j*mesh_w:(j+1)*mesh_w]
                f_k = f_k.reshape(feature.size(0), -1, opt.represent_feature)

                attention = torch.bmm(f_k, r_k) / torch.sqrt(torch.tensor(opt.represent_feature))
                attention = self.act(attention)
                mesh = torch.bmm(attention, t_k.transpose(1, 2))
                mesh = mesh.transpose(1, 2)
                Y_L[:, :, i * mesh_h:(i + 1) * mesh_h, j * mesh_w:(j + 1) * mesh_w] = mesh.reshape(mesh.size(0), 3, mesh_h, mesh_w)

        return pad_tensor_back(Y_L, pad_left, pad_right, pad_top, pad_bottom)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        relu5_3 = F.relu(self.conv5_3(relu5_2), inplace=True)

        if opt.vgg_choose == "conv4_3":
            return relu4_3
        elif opt.vgg_choose == "conv4_2":
            return relu4_2
        elif opt.vgg_choose == "conv5_3":
            return relu5_3
        elif opt.vgg_choose == "conv5_2":
            return relu5_2
        elif opt.vgg_choose == "conv5_1":
            return relu5_1


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(8, 3, 256, 256).to(device)
    fea = torch.randn(8, 16, 400, 600).to(device)

    from options import TrainOptions
    opt = TrainOptions().parse()

    encoder = define_encoder(opt, device)
    bifpn = define_bifpn(opt, device)
    global_rct = define_G_rct(opt, device)
    local_rct = define_L_rct(opt, device)

    features = encoder(x)
    # print(features[1].shape)

    fusion_features = bifpn(features)
    # print(fusion_features[0].shape)

    Y_G = global_rct(fea, fusion_features[3])
    print(Y_G.shape)

    Y_L = local_rct(fea, fusion_features[0])
    print(Y_L.shape)
