import os
import torch
from torch import nn
import networks  # 调试完改为from ... import
import loss_function as Loss  # 同上
from models.base_model import BaseModel


class RCTNet(BaseModel):
    def name(self):
        return 'RCTNet'

    def __init__(self, opt):
        super(RCTNet, self).__init__(opt)

        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.isTrain = opt.isTrain
        self.input_A = torch.zeros([opt.batch_size, 3, opt.fine_size, opt.fine_size], device=self.device)
        self.input_B = torch.zeros([opt.batch_size, 3, opt.fine_size, opt.fine_size], device=self.device)
        self.input_img = torch.zeros([opt.batch_size, 3, opt.fine_size, opt.fine_size], device=self.device)
        self.img_paths = []

        # Definition of loss function
        self.l1_criterion = nn.L1Loss()
        self.l1_criterion.to(self.device)
        self.vgg_criterion = Loss.PerceptualLoss(opt)
        self.vgg_criterion.to(self.device)
        self.vgg = Loss.load_vgg16("./")
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.l1_loss = 0
        self.vgg_loss = 0
        self.loss = 0

        # Definition of Network
        self.encoder, param = networks.define_encoder(self.opt, self.device)
        self.params = param
        self.bifpn, param = networks.define_bifpn(self.opt, self.device)
        self.params += param
        self.global_rct, param = networks.define_G_rct(self.opt, self.device)
        self.params += param
        self.local_rct, param = networks.define_L_rct(self.opt, self.device)
        self.params += param
        self.extract_f, param = networks.define_feature(self.opt, self.device)
        self.params += param
        self.fusion = networks.Fusion().to(self.device)
        self.params += [{'params': list(self.fusion.parameters())}]

        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.params,
                lr=self.opt.lr,
                betas=(self.opt.beta1, 0.999),
                weight_decay=1e-5
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

            self.encoder.train()
            self.bifpn.train()
            self.global_rct.train()
            self.local_rct.train()
            self.extract_f.train()
        else:
            which_epoch = self.opt.which_epoch
            self.load_network(self.encoder, 'E', which_epoch)
            self.load_network(self.bifpn, 'B', which_epoch)
            self.load_network(self.global_rct, 'G', which_epoch)
            self.load_network(self.local_rct, 'L', which_epoch)
            self.load_network(self.extract_f, 'F', which_epoch)

            self.encoder.eval()
            self.bifpn.eval()
            self.global_rct.eval()
            self.local_rct.eval()
            self.extract_f.eval()

        print('---------- Networks initialized -------------')
        networks.print_network(self.encoder)
        networks.print_network(self.bifpn)
        networks.print_network(self.global_rct)
        networks.print_network(self.local_rct)
        networks.print_network(self.extract_f)
        print('-----------------------------------------------')

    def set_input(self, data):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = data['A' if AtoB else 'B']
        input_B = data['B' if AtoB else 'A']
        input_img = data['input']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.img_paths = data['A_paths' if AtoB else 'B_paths']

    # get image paths
    def get_image_paths(self):
        return self.img_paths

    def forward(self):
        org_feature = self.extract_f(self.input_img)
        features = self.encoder(self.input_A)
        fusion_features = self.bifpn(features)

        Y_G = self.global_rct(org_feature, fusion_features[3])
        Y_L = self.local_rct(org_feature, fusion_features[0])

        return self.fusion(Y_G, Y_L)

    def _backward(self):
        Y = self.forward()

        self.l1_loss = self.l1_criterion(Y, self.input_img)
        self.vgg_loss = self.vgg_criterion.compute_vgg_loss(self.vgg, Y, self.input_img)
        self.loss = self.l1_loss + self.opt.balance_lambda * self.vgg_loss

        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self._backward()
        self.optimizer.step()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    a = torch.randn(8, 3, 256, 256).to(device)
    b = torch.randn(8, 3, 256, 256).to(device)
    x_org = torch.randn(8, 3, 400, 600).to(device)
    a_path = []
    b_path = []
    img_group = {'A': a, 'B': b, 'input': x_org, 'A_paths': a_path, 'B_paths': b_path}

    from options import TrainOptions
    opt = TrainOptions().parse()

    model = RCTNet(opt)

    model.set_input(img_group)

    # f = model.forward()
    # print(f.shape)

    model.optimize_parameters()

    pass
