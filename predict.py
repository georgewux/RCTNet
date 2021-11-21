import os
import time
from options import TestOptions
from custom_dataloader import CustomDataLoader
from utils.visualizer import Visualizer
from utils import html


def create_model(opt):
    model = None
    if opt.model == 'rct_net':
        assert (opt.dataset_mode == 'pair')
        from models.rctnet_model import RCTNet
        model = RCTNet(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    print("model [%s] was created" % (model.name()))
    return model


opt = TestOptions().parse()
opt.batch_size = 1
opt.no_flip = True

data_loader = CustomDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# 输出融合权重
# param = model.params
# print(param[0]['params'][0])

web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print(len(dataset))
for i, data in enumerate(dataset):
    model.set_input(data)
    visuals = model.predict()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
