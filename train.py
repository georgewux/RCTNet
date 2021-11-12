import torch
import time
from options import TrainOptions
from custom_dataloader import CustomDataLoader
from utils.visualizer import Visualizer


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


def main():
    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameters Configuration
    opt = TrainOptions().parse()

    # DataLoader
    data_loader = CustomDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)

    for epoch in range(1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            model.set_input(data)


if __name__ == '__main__':
    main()
