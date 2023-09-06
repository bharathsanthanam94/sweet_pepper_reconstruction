#!/usr/bin/env python3

import csv
import numpy as np
import os
from collections import OrderedDict
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from natsort import natsorted
import imageio

from ipb_loaders.ipb_base import IPB_Base


class ScanNet(IPB_Base):
    """ScanNet dataset http://www.scan-net.org/
    Keyword arguments:
    - root_dir (``string``): Path to the base directory of the dataset
    - scene_file (``string``): Path to file containing a list of scenes to be loaded
    - transform (``callable``, optional): A function/transform that takes in a 
    PIL image and returns a transformed version of the image. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes 
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its path.
    By default, ``default_loader`` is used.
    - color_mean (``list``): A list of length 3, containing the R, G, B channelwise mean.
    - color_std (``list``): A list of length 3, containing the R, G, B channelwise standard deviation.
    - load_depth (``bool``): Whether or not to load depth images (architectures that use depth 
    information need depth to be loaded).
    - seg_classes (``string``): The palette of classes that the network should learn.
    """
    def __init__(self,
                 data_source=None,
                 split='train',
                 transform=None,
                 label_transform=None,
                 color_mean=[0., 0., 0.],
                 color_std=[1., 1., 1.],
                 load_depth=True,
                 seg_classes='nyu40',
                 path_in_dataserver='/export/datasets/ScanNet/',
                 unittesting=False):
        super().__init__(data_source, path_in_dataserver, unittesting)

        self.root_dir = os.path.join(data_source, 'scans')
        self.scene_file = os.path.join(data_source, split + '.txt')
        self.mode = split
        self.transform = transform
        self.label_transform = label_transform
        self.loader = scannet_loader
        self.length = 0
        self.color_mean = color_mean
        self.color_std = color_std
        self.load_depth = load_depth
        self.seg_classes = seg_classes
        self.setup_files = self.data_source

        self.prepare_labels()
        # color_encoding has to be initialized AFTER seg_classes
        self.color_encoding = self.get_color_encoding()

        if self.load_depth is True:
            self.loader = scannet_loader_depth

        # Get the list of scenes, and generate paths
        scene_list = []

        try:
            scene_file = open(self.scene_file, 'r')
            scenes = scene_file.readlines()
            scene_file.close()
            for scene in scenes:
                scene = scene.strip().split()
                scene_list.append(scene[0])
        except Exception as e:
            raise e

        self.resize = transforms.Resize((480, 640), InterpolationMode.NEAREST)

        if self.mode.lower() == 'train':
            # Get train data and labels filepaths
            self.train_data = []
            self.train_depth = []
            self.train_labels = []
            for scene in scene_list:
                color_images, depth_images, labels = get_filenames_scannet(self.root_dir, scene)
                self.train_data += color_images
                self.train_depth += depth_images
                self.train_labels += labels
                self.length += len(color_images)
        elif self.mode.lower() == 'val':
            # Get val data and labels filepaths
            self.val_data = []
            self.val_depth = []
            self.val_labels = []
            for scene in scene_list:
                color_images, depth_images, labels = get_filenames_scannet(self.root_dir, scene)
                self.val_data += color_images
                self.val_depth += depth_images
                self.val_labels += labels
                self.length += len(color_images)
        elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
            # Get test data and labels filepaths
            self.test_data = []
            self.test_depth = []
            self.test_labels = []
            for scene in scene_list:
                color_images, depth_images, labels = get_filenames_scannet(self.root_dir, scene)
                self.test_data += color_images
                self.test_depth += depth_images
                self.test_labels += labels
                self.length += len(color_images)
        else:
            raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test, inference')

    def __getitem__(self, index):
        """ Returns element at index in the dataset.
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth of the image
        """

        sample = {}
        if self.load_depth is True:

            if self.mode.lower() == 'train':
                data_path, depth_path, label_path = self.train_data[index], self.train_depth[index], \
                    self.train_labels[index]
            elif self.mode.lower() == 'val':
                data_path, depth_path, label_path = self.val_data[index], self.val_depth[index], \
                    self.val_labels[index]
            elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
                data_path, depth_path, label_path = self.test_data[index], self.test_depth[index], \
                    self.test_labels[index]
            else:
                raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test, inference')

            rgbd, label = self.loader(data_path, depth_path, label_path, self.color_mean, self.color_std, self.seg_classes)

            label = self.rescale_labels(label)
            label = torch.from_numpy(label)
            label = self.resize(label.unsqueeze(0)).squeeze().long()

            if self.mode.lower() == 'inference':
                sample['image'] = rgbd
                sample['semantic'] = label
                sample['filename'] = data_path
            else:
                sample['image'] = rgbd
                sample['semantic'] = label
            # import ipdb;ipdb.set_trace()  # fmt: skip
            return sample

        else:

            if self.mode.lower() == 'train':
                data_path, label_path = self.train_data[index], self.train_labels[index]
            elif self.mode.lower() == 'val':
                data_path, label_path = self.val_data[index], self.val_labels[index]
            elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
                data_path, label_path = self.test_data[index], self.test_labels[index]
            else:
                raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test')

            img, label = self.loader(data_path, label_path, self.color_mean, self.color_std, self.seg_classes)

            if self.mode.lower() == 'inference':
                sample['image'] = img
                sample['semantic'] = label
                sample['filename'] = data_path
            else:
                sample['image'] = img
                sample['semantic'] = label
            return sample

    def __len__(self):
        """ Returns the length of the dataset. """
        return self.length

    def rescale_labels(self, semantic):
        labels = np.unique(semantic)
        for index in labels:
            if index not in self.labels_mapping:
                semantic[semantic == index] = 0
            else:
                mapped_label = self.labels_mapping[index]
                semantic[semantic == index] = self.labels_numbers.index(mapped_label)
        return semantic

    def prepare_labels(self):
        self.labels_numbers = []
        labels_names = []
        nyu_labels = os.path.join(self.setup_files, "nyu_labels.txt")
        with open(nyu_labels) as f:
            fs = csv.reader(f, delimiter=" ")
            for line in fs:
                self.labels_numbers.append(int(line[0]))
                labels_names.append(line[1])

        self.labels_mapping = {}
        scannet_tsv = os.path.join(self.setup_files, "scannetv2-labels.combined.tsv")
        with open(scannet_tsv) as f:
            rd = csv.reader(f, delimiter="\t")
            for i, row in enumerate(rd):
                if i > 0:
                    scannet_label = int(row[0])
                    nyu_label = int(row[4])
                    if nyu_label in self.labels_numbers:
                        self.labels_mapping[scannet_label] = nyu_label

    def get_color_encoding(self):
        if self.seg_classes.lower() == 'nyu40':
            """Color palette for nyu40 labels """
            return OrderedDict([
                ('unlabeled', (0, 0, 0)),
                ('wall', (174, 199, 232)),
                ('floor', (152, 223, 138)),
                ('cabinet', (31, 119, 180)),
                ('bed', (255, 187, 120)),
                ('chair', (188, 189, 34)),
                ('sofa', (140, 86, 75)),
                ('table', (255, 152, 150)),
                ('door', (214, 39, 40)),
                ('window', (197, 176, 213)),
                ('bookshelf', (148, 103, 189)),
                ('picture', (196, 156, 148)),
                ('counter', (23, 190, 207)),
                ('blinds', (178, 76, 76)),
                ('desk', (247, 182, 210)),
                ('shelves', (66, 188, 102)),
                ('curtain', (219, 219, 141)),
                ('dresser', (140, 57, 197)),
                ('pillow', (202, 185, 52)),
                ('mirror', (51, 176, 203)),
                ('floormat', (200, 54, 131)),
                ('clothes', (92, 193, 61)),
                ('ceiling', (78, 71, 183)),
                ('books', (172, 114, 82)),
                ('refrigerator', (255, 127, 14)),
                ('television', (91, 163, 138)),
                ('paper', (153, 98, 156)),
                ('towel', (140, 153, 101)),
                ('showercurtain', (158, 218, 229)),
                ('box', (100, 125, 154)),
                ('whiteboard', (178, 127, 135)),
                ('person', (120, 185, 128)),
                ('nightstand', (146, 111, 194)),
                ('toilet', (44, 160, 44)),
                ('sink', (112, 128, 144)),
                ('lamp', (96, 207, 209)),
                ('bathtub', (227, 119, 194)),
                ('bag', (213, 92, 176)),
                ('otherstructure', (94, 106, 211)),
                ('otherfurniture', (82, 84, 163)),
                ('otherprop', (100, 85, 144)),
            ])
        elif self.seg_classes.lower() == 'scannet20':
            return OrderedDict([
                ('unlabeled', (0, 0, 0)),
                ('wall', (174, 199, 232)),
                ('floor', (152, 223, 138)),
                ('cabinet', (31, 119, 180)),
                ('bed', (255, 187, 120)),
                ('chair', (188, 189, 34)),
                ('sofa', (140, 86, 75)),
                ('table', (255, 152, 150)),
                ('door', (214, 39, 40)),
                ('window', (197, 176, 213)),
                ('bookshelf', (148, 103, 189)),
                ('picture', (196, 156, 148)),
                ('counter', (23, 190, 207)),
                ('desk', (247, 182, 210)),
                ('curtain', (219, 219, 141)),
                ('refrigerator', (255, 127, 14)),
                ('showercurtain', (158, 218, 229)),
                ('toilet', (44, 160, 44)),
                ('sink', (112, 128, 144)),
                ('bathtub', (227, 119, 194)),
                ('otherfurniture', (82, 84, 163)),
            ])


def get_filenames_scannet(base_dir, scene_id):
    """Helper function that returns a list of scannet images and the corresponding 
	segmentation labels, given a base directory name and a scene id.

	Args:
	- base_dir (``string``): Path to the base directory containing ScanNet data, in the 
	directory structure specified in https://github.com/angeladai/3DMV/tree/master/prepare_data
	- scene_id (``string``): ScanNet scene id

	"""

    if not os.path.isdir(base_dir):
        raise RuntimeError('\'{0}\' is not a directory.'.format(base_dir))

    color_images = []
    depth_images = []
    labels = []

    # Explore the directory tree to get a list of all files
    for path, _, files in os.walk(os.path.join(base_dir, scene_id, 'color')):
        files = natsorted(files)
        for file in files:
            filename, _ = os.path.splitext(file)
            depthfile = os.path.join(base_dir, scene_id, 'depth', filename + '.png')
            labelfile = os.path.join(base_dir, scene_id, 'label-filt', str(int(filename)) + '.png')
            # Add this file to the list of train samples, only if its corresponding depth and label
            # files exist.
            if os.path.exists(depthfile) and os.path.exists(labelfile):
                color_images.append(os.path.join(base_dir, scene_id, 'color', filename + '.jpg'))
                depth_images.append(depthfile)
                labels.append(labelfile)

    # Assert that we have the same number of color, depth images as labels
    assert (len(color_images) == len(depth_images) == len(labels))

    return color_images, depth_images, labels


def scannet_loader(data_path, label_path, color_mean=[0., 0., 0.], color_std=[1., 1., 1.], seg_classes='nyu40'):
    """Loads a sample and label image given their path as PIL images. (nyu40 classes)

	Keyword arguments:
	- data_path (``string``): The filepath to the image.
	- label_path (``string``): The filepath to the ground-truth image.
	- color_mean (``list``): R, G, B channel-wise mean
	- color_std (``list``): R, G, B channel-wise stddev
	- seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')

	Returns the image and the label as PIL images.

	"""

    # Load image
    data = np.array(imageio.imread(data_path))
    # Reshape data from H x W x C to C x H x W
    data = np.moveaxis(data, 2, 0)
    # Define normalizing transform
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    # Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
    data = normalize(torch.Tensor(data.astype(np.float32) / 255.0))

    # Load label
    if seg_classes.lower() == 'nyu40':
        label = np.array(imageio.imread(label_path)).astype(np.uint8)
    elif seg_classes.lower() == 'scannet20':
        label = np.array(imageio.imread(label_path)).astype(np.uint8)
        # Remap classes from 'nyu40' to 'scannet20'
        label = nyu40_to_scannet20(label)

    return data / 255, label


def scannet_loader_depth(data_path, depth_path, label_path, color_mean=[0.,0.,0.], color_std=[1.,1.,1.], \
 seg_classes='nyu40'):
    """Loads a sample and label image given their path as PIL images. (nyu40 classes)

	Keyword arguments:
	- data_path (``string``): The filepath to the image.
	- depth_path (``string``): The filepath to the depth png.
	- label_path (``string``): The filepath to the ground-truth image.
	- color_mean (``list``): R, G, B channel-wise mean
	- color_std (``list``): R, G, B channel-wise stddev
	- seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')

	Returns the image and the label as PIL images.

	"""

    # Load image
    rgb = np.array(imageio.imread(data_path))
    # Reshape rgb from H x W x C to C x H x W
    rgb = np.moveaxis(rgb, 2, 0)
    # Define normalizing transform
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    # Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
    rgb = normalize(torch.Tensor(rgb.astype(np.float32) / 255.0))

    # Load depth
    depth = torch.Tensor(np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0)
    depth = torch.unsqueeze(depth, 0)

    # Concatenate rgb and depth
    data = torch.cat((rgb, depth), 0)

    # Load label
    if seg_classes.lower() == 'nyu40':
        label = np.array(imageio.imread(label_path)).astype(np.uint8)
    elif seg_classes.lower() == 'scannet20':
        label = np.array(imageio.imread(label_path)).astype(np.uint8)

    return data / 255, label


def nyu40_to_scannet20(label):
    """Remap a label image from the 'nyu40' class palette to the 'scannet20' class palette """

    # Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29. 30. 31. 32, 35. 37. 38, 40
    # Because, these classes from 'nyu40' are absent from 'scannet20'. Our label files are in
    # 'nyu40' format, hence this 'hack'. To see detailed class lists visit:
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt ('nyu40' labels)
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt ('scannet20' labels)
    # The remaining labels are then to be mapped onto a contiguous ordering in the range [0,20]

    # The remapping array comprises tuples (src, tar), where 'src' is the 'nyu40' label, and 'tar' is the
    # corresponding target 'scannet20' label
    remapping = [(0, 0), (13, 0), (15, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (25, 0), (26, 0), (27, 0), (29, 0), (30, 0), (31, 0),
                 (32, 0), (35, 0), (37, 0), (38, 0), (40, 0), (14, 13), (16, 14), (24, 15), (28, 16), (33, 17), (34, 18), (36, 19), (39, 20)]
    for src, tar in remapping:
        label[np.where(label == src)] = tar
    return label
