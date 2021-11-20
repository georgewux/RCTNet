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
    # Parameters Configuration
    opt = TrainOptions().parse()

    # DataLoader
    data_loader = CustomDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)

    total_steps = 0

    for epoch in range(1, opt.num_epoch + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors(epoch)
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.num_epoch, time.time() - epoch_start_time))


if __name__ == '__main__':
    main()
