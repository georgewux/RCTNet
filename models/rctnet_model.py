import torch
from torch import nn
from utils import util
from models import networks  # 调试完改为from ... import
from models import loss_function as Loss  # 同上
from models.base_model import BaseModel
from collections import OrderedDict


class RCTNet(BaseModel):
    def name(self):
        return 'RCTNet'

    def __init__(self, opt):
        super(RCTNet, self).__init__(opt)

        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.isTrain = opt.isTrain
        self.input_A = torch.zeros([opt.batch_size, 3, opt.fine_size, opt.fine_size], device=self.device)
        self.input_img = torch.zeros([opt.batch_size, 3, opt.fine_size, opt.fine_size], device=self.device)
        self.target_img = torch.zeros([opt.batch_size, 3, opt.fine_size, opt.fine_size], device=self.device)
        self.img_paths = []

        # Definition of loss function
        self.l1_criterion = nn.L1Loss()
        self.l1_criterion.to(self.device)
        self.vgg_criterion = Loss.PerceptualLoss(opt)
        self.vgg_criterion.to(self.device)
        self.vgg = Loss.load_vgg16("./models", self.device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Definition of Network
        self.fusion, self.params = networks.define_fusion(self.opt, self.device)
        # print(self.params)

        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.params,
                lr=self.opt.lr,
                betas=(self.opt.beta1, 0.999)
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

            self.fusion.train()
        else:
            which_epoch = self.opt.which_epoch
            self.load_network(self.fusion, 'F', which_epoch)

            self.fusion.eval()

        print('---------- Networks initialized -------------')
        networks.print_network(self.fusion)
        print('-----------------------------------------------')

    def set_input(self, data):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = data['A' if AtoB else 'B']
        input_img = data['input']
        target_img = data['target']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.target_img.resize_(target_img.size()).copy_(target_img)
        self.img_paths = data['A_paths' if AtoB else 'B_paths']

    def predict(self):
        pass

    # get image paths
    def get_image_paths(self):
        return self.img_paths

    def forward(self):
        self.Y = self.fusion(self.input_img, self.input_A)

    def backward(self):
        self.l1_loss = self.l1_criterion(self.Y, self.target_img)
        self.vgg_loss = self.vgg_criterion.compute_vgg_loss(self.vgg, self.Y, self.target_img)
        self.loss = self.l1_loss + self.opt.balance_lambda * self.vgg_loss

        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_visuals(self):
        input_img = util.tensor2im(self.input_img.data)
        target_img = util.tensor2im(self.target_img.data)
        enhanced_img = util.tensor2im(self.Y.data)
        return OrderedDict([('input_img', input_img), ('target_img', target_img), ('enhanced_img', enhanced_img)])

    def get_current_errors(self, epoch):
        l1 = self.l1_loss.item()
        vgg = self.opt.balance_lambda * self.vgg_loss.item()
        return OrderedDict([('l1', l1), ("vgg", vgg)])

    def save(self, label):
        self.save_network(self.fusion, 'F', label, self.device)

    def update_learning_rate(self):
        self.scheduler.step()


if __name__ == '__main__':
    device = torch.device('cpu')

    a = torch.randn(8, 3, 256, 256).to(device)
    x_org = torch.randn(8, 3, 400, 600).to(device)
    x_tar = torch.randn(8, 3, 400, 600).to(device)

    a_path = []
    b_path = []
    img_group = {'A': a, 'input': x_org, 'target': x_tar, 'A_paths': a_path, 'B_paths': b_path}

    from options import TrainOptions
    opt = TrainOptions().parse()

    model = RCTNet(opt)

    model.set_input(img_group)

    model.optimize_parameters()

