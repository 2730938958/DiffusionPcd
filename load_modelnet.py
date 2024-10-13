from loader import make_loader
from arguments import rpmnet_train_arguments
from data_loader.datasets import get_train_datasets
parser = rpmnet_train_arguments()
_args = parser.parse_args()

train_set, val_set = get_train_datasets(_args)
train_loader, val_loader = make_loader(train_set,val_set, 4, 4)
for epoch in range(0, _args.epochs):
    for train_data in train_loader:
        points = train_data['input_lidar']