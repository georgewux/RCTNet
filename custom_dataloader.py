from torch.utils.data import DataLoader


def create_dataset(opt):
    dataset = None
    if opt.dataset_mode == 'single':
        from datasets.single_dataset import SingleDataset
        dataset = SingleDataset(opt)
    elif opt.dataset_mode == 'pair':
        from datasets.pair_dataset import PairDataset
        dataset = PairDataset(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % dataset.name())
    return dataset


class CustomDataLoader(object):
    def __init__(self, opt):
        self.dataset = create_dataset(opt)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.isTrain,
            num_workers=opt.num_workers,
            pin_memory=True
        )

    def load_data(self):
        return self.dataloader
