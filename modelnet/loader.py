import torch
from data_loader.datasets import ModelNetHdf
import torch.utils.data
import argparse
import torchvision
from arguments import rpmnet_train_arguments
import torch
parser = rpmnet_train_arguments()
_args = parser.parse_args()



def get_train_datasets(args: argparse.Namespace):
    train_categories, val_categories = None, None
    if args.train_categoryfile:
        train_categories = [line.rstrip('\n') for line in open(args.train_categoryfile)]
        train_categories.sort()
    if args.val_categoryfile:
        val_categories = [line.rstrip('\n') for line in open(args.val_categoryfile)]
        val_categories.sort()


    if args.dataset_type == 'modelnet_hdf':
        train_data = ModelNetHdf(args.dataset_path, subset='train', categories=train_categories)
        val_data = ModelNetHdf(args.dataset_path, subset='test', categories=val_categories)
    else:
        raise NotImplementedError

    return train_data, val_data

def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    batch_data = {
        'input_lidar': torch.nn.utils.rnn.pad_sequence([torch.tensor(sample['points'][:, :3].copy()) for sample in batch]).permute(1,0,2)
    }

    return batch_data

def make_loader(train_set, val_set, train_batch_size, val_batch_size, num_workers, rng_generator):

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set,pin_memory=False,collate_fn=collate_fn_padd,drop_last=True,generator=rng_generator,
                                               batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set,pin_memory=False,collate_fn=collate_fn_padd,drop_last=False,generator=rng_generator,
                                             batch_size=val_batch_size, shuffle=False, num_workers=num_workers)


    return train_loader, val_loader
    # for epoch in range(0, _args.epochs):
    #     for train_data in train_loader:
    #         points = train_data['input_lidar']

