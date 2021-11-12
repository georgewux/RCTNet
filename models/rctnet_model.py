import os
import torch
from torch import nn
from models import networks
from models import loss_function as Loss
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

        self.l1_loss = nn.L1Loss()
        self.l1_loss.to(self.device)
        self.vgg_loss = Loss.PerceptualLoss(opt)
        self.vgg_loss.to(self.device)
        self.vgg = Loss.load_vgg16("./models")
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.encoder = networks.define_encoder(self.opt, self.device)
        self.bifpn = networks.define_bifpn(self.opt, self.device)
        self.global_rct = networks.define_G_rct(self.opt, self.device)
        self.local_rct = networks.define_L_rct(self.opt, self.device)

        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.bifpn.parameters()) + list(
                    self.global_rct.parameters()) + list(self.local_rct.parameters()),
                lr=self.opt.lr,
                betas=(self.opt.beta1, 0.999),
                weight_decay=1e-5
            )

        if opt.isTrain:
            self.encoder.train()
            self.bifpn.train()
            self.global_rct.train()
            self.local_rct.train()
        else:
            self.encoder.eval()
            self.bifpn.eval()
            self.global_rct.eval()
            self.local_rct.eval()

        print('---------- Networks initialized -------------')
        networks.print_network(self.encoder)
        networks.print_network(self.bifpn)
        networks.print_network(self.global_rct)
        networks.print_network(self.local_rct)
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
