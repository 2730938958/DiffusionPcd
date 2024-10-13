"""Data loader
"""
import argparse
import logging
import os
from typing import List
import h5py
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torchvision
import torch
import data_loader.transforms as Transforms
import common.math.se3 as se3
from scipy.linalg import expm, norm
from glob import glob

_logger = logging.getLogger()


def get_train_datasets(args: argparse.Namespace):
    train_categories, val_categories = None, None
    if args.train_categoryfile:
        train_categories = [line.rstrip('\n') for line in open(args.train_categoryfile)]
        train_categories.sort()
    if args.val_categoryfile:
        val_categories = [line.rstrip('\n') for line in open(args.val_categoryfile)]
        val_categories.sort()

    # train_transforms, val_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
    #                                                   args.num_points, args.partial)
    # _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
    # _logger.info('Val transforms: {}'.format(', '.join([type(t).__name__ for t in val_transforms])))
    # train_transforms = torchvision.transforms.Compose(train_transforms)
    # val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.dataset_type == 'modelnet_hdf':
        train_data = ModelNetHdf(args.dataset_path, subset='train', categories=train_categories)
        val_data = ModelNetHdf(args.dataset_path, subset='test', categories=val_categories)

    elif args.dataset_type == 'cs':
        benchmark_path = "D:\\AI\\CSdataset\\cross-source-dataset"
        train_data = CrosssourceDataset(benchmark_path, transform=train_transforms)
        val_data = CrosssourceDataset(benchmark_path, transform=val_transforms)

    else:
        raise NotImplementedError

    return train_data, val_data


def get_test_datasets(args: argparse.Namespace):
    test_categories = None
    if args.test_category_file:
        test_categories = [line.rstrip('\n') for line in open(args.test_category_file)]
        test_categories.sort()

    _, test_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                        args.num_points, args.partial)
    _logger.info('Test transforms: {}'.format(', '.join([type(t).__name__ for t in test_transforms])))
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.dataset_type == 'modelnet_hdf':
        test_data = ModelNetHdf(args.dataset_path, subset='test', categories=test_categories,
                                transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms





class CrosssourceDataset(torch.utils.data.Dataset):
    """
    Data loader class to read cross-source point clouds
    """
    def __init__(self, data_path, transform):
        """ initialize the dataloader. get all file names"""
        self.names = self.find_pairs(data_path)
        self.randg = np.random.RandomState()
        self.rotation_range = 180  # rotation
        self._transform = transform

    def __getitem__(self, item):
        """ get data item """
        pair_path = self.names[item]
        # read point cloud data
        if "kinect_sfm" in pair_path:
            kinect_path = os.path.join(self.names[item], "kinect.ply")
            pc_kinect = o3d.io.read_point_cloud(kinect_path)
            sfm_path = os.path.join(self.names[item], "sfm.ply")
            pc_sfm = o3d.io.read_point_cloud(sfm_path)

            xyz0 = np.array(pc_kinect.points)
            xyz1 = np.array(pc_sfm.points)

            # scaled the point clouds to [-1 1]
            centroid = np.mean(xyz0, axis=0)
            xyz0 -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz0) ** 2, axis=-1)))
            xyz0 /= furthest_distance

            centroid = np.mean(xyz1, axis=0)
            xyz1 -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz1) ** 2, axis=-1)))
            xyz1 /= furthest_distance

        elif "kinect_lidar" in pair_path:
            kinect_path = os.path.join(self.names[item], "kinect.ply")
            pc_kinect = o3d.io.read_point_cloud(kinect_path)
            sfm_path = os.path.join(self.names[item], "lidar.ply")
            pc_sfm = o3d.io.read_point_cloud(sfm_path)

            xyz0 = np.array(pc_kinect.points)
            xyz1 = np.array(pc_sfm.points)

            # scaled the point clouds to [-1 1]
            centroid = np.mean(xyz0, axis=0)
            xyz0 -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz0) ** 2, axis=-1)))
            xyz0 /= furthest_distance

            centroid = np.mean(xyz1, axis=0)
            xyz1 -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz1) ** 2, axis=-1)))
            xyz1 /= furthest_distance
        # read ground truth transformation
        T_gt = np.loadtxt(os.path.join(self.names[item], "T_gt.txt"))

        return xyz0, xyz1, T_gt

    def __len__(self):
        """
        calculate sample number of the dataset
        :return:
        """
        num = len(self.names)
        return num

    def find_pairs(self, path):
        """
        # find all the ply filenames in the given path
        :param path: given data set directory
        :return:
        """
        easy_root = os.path.join(path, "kinect_sfm/easy/*/")
        subfolders_easy = glob(easy_root)
        hard_root = os.path.join(path, "kinect_sfm/hard/*/")
        subfolders_hard = glob(hard_root)
        kinect_lidar_root = os.path.join(path, "kinect_lidar/*/*/")
        subfolders_kinect_lidar = glob(kinect_lidar_root)
        subfolders = subfolders_easy + subfolders_hard + subfolders_kinect_lidar
        return subfolders


    def apply_transform(self, pts, trans):
        """
        # apply transformation to a point cloud
        :param pts: N x 3
        :param trans: 4 x 4
        :return: N X 3
        """
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def M(self, axis, theta):
        """
        # Genearte rotation matrix along axis with angle theta
        :param axis: axis number [0, 1, 2] means [x, y, z]
        :param theta: rotation angle [-3.14, 3.14]
        :return: rotation matrix [3 x 3]
        """
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def sample_random_trans(self, pcd, randg, rotation_range=360):
        """
        # generate a trasnformation matrix with Random rotation
        :param pcd: input point cloud, numpy
        :param randg: random method
        :param rotation_range: random rotation angle [-180, 180]
        :return: a transformation matrix [4 x 4]
        """
        T = np.eye(4)
        R = self.M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
        T[:3, :3] = R
        T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
        return T


class ModelNetHdf(Dataset):
    def __init__(self, dataset_path: str, subset: str = 'train', categories: List = None, transform=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available

        Args:
            dataset_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path

        metadata_fpath = os.path.join(self._root, '{}_files.txt'.format(subset))
        self._logger.info('Loading data from {} for {}'.format(metadata_fpath, subset))

        if not os.path.exists(os.path.join(dataset_path)):
            self._download_dataset(dataset_path)

        with open(os.path.join(dataset_path, 'shape_names.txt')) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        with open(os.path.join(dataset_path, '{}_files.txt'.format(subset))) as fid:
            h5_filelist = [line.strip() for line in fid]
            h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
            h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._logger.info('Categories used: {}.'.format(categories_idx))
            self._classes = categories
        else:
            categories_idx = None
            self._logger.info('Using all categories.')

        self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
        # self._data, self._labels = self._data[:32], self._labels[:32, ...]
        self._transform = transform
        self._logger.info('Loaded {} {} instances.'.format(self._data.shape[0], subset))

    def __getitem__(self, item):
        sample = {'points': self._data[item, :, :], 'label': self._labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._data.shape[0]

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_h5_files(fnames, categories):

        all_data = []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    @staticmethod
    def _download_dataset(dataset_path: str):
        os.makedirs(dataset_path, exist_ok=True)

        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget {}'.format(www))
        os.system('unzip {} -d .'.format(zipfile))
        os.system('mv {} {}'.format(zipfile[:-4], os.path.dirname(dataset_path)))
        os.system('rm {}'.format(zipfile))

    def to_category(self, i):
        return self._idx2category[i]
