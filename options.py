import os
import argparse
from utils import util


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        # Basic argument
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='rct_net',
                                 help='chooses which model to use. rct_net, enlighten_gan, test')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # About Data process
        self.parser.add_argument('--data_root', default='./datasets/LoL/train',
                                 help='path of images(should have subfolders)')
        self.parser.add_argument('--dataset_mode', type=str, default='pair',
                                 help='chooses how datasets are loaded. [pair | single]')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--batch_size', default=8, type=int, help='the size of the batches')
        self.parser.add_argument('--num_workers', default=0, type=int, help='number of workers while load data')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize',
                                 help='scaling and cropping of images at load time')
        self.parser.add_argument('--fine_size', default=256, type=int, help='crop to this size')
        self.parser.add_argument('--scale_width', default=512, type=int, help='scale to this size')
        self.parser.add_argument('--no_flip', action='store_true', help='do not flip the images for data augmentation')
        self.parser.add_argument('--color_jitter', action='store_true', help='verify the light for data augmentation')

        # Network
        self.parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay without BatchNorm')
        self.parser.add_argument('--num_filter', default=16, type=int, help='number of filters in first conv layer')
        self.parser.add_argument('--fusion_filter', default=128, type=int, help='number of filters in fusion layer')
        self.parser.add_argument('--represent_feature', default=16, type=int, help='number of representative features')
        self.parser.add_argument('--ngf', default=64, type=int, help='number of global rct feature')
        self.parser.add_argument('--nlf', default=16, type=int, help='number of local rct feature')
        self.parser.add_argument('--mesh_size', default=31, type=int, help='number of mesh for local rct')
        self.parser.add_argument('--balance_lambda', default=0.04, type=float, help='weight to balance two loss')

        # perceptual loss from vgg
        self.parser.add_argument('--vgg_mean', action='store_true', help='substract mean in vgg loss')


        # Visualization
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # 在opt中声明一个新变量，并将子类中的变量值赋给它

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [checkpoints_dir]/[name]/web/')
        self.parser.add_argument('--display_freq', type=int, default=30,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')

        self.parser.add_argument('--lr', type=float, default=5e-4, help='learning rate of training')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--num_epoch', type=int, default=100, help='number of epoch during training')

        self.isTrain = True


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        self.isTrain = False
