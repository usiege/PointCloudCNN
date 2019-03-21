import os
import numpy as np

from easydict import EasyDict as edict

import argparse

# flags for networks
def arg_config():
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-n', '--net', default='pointnet',
						help='pointcloud net use ["pointnet","squeezeseg","voxelnet"]')
	
	parser.add_argument('--p', default='train', help='for process use ["train","eval","test"]')
	parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
	parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
	parser.add_argument('--num_point', type=int, default=4096 * 8, help='Point number [default: 4096]')
	parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 50]')
	parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 24]')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
	parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
	parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
	parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
	parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
	parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
	
	return parser.parse_args()


# base config for networks
def base_model_config(dataset=''):

	cfg = edict()

	# imageset files names
	cfg.IMAGESET = ['all', 'test', 'train', 'eval']

	# Dataset used to train/val/test model.
	cfg.DATASET = dataset.upper()

	# data path
	cfg.DATA_PATH = ''
	
	# imageset path
	cfg.IMAGESET_PATH = ''

	# classes
	cfg.CLASSES = ['unknown', 'car', 'pedestrian', 'cyclist']
	
	# loss weight
	cfg.CLS_LOSS_WEIGHT = [1.0/15.0, 1.0, 10.0, 10.0]
	
	# number of classes
	cfg.NUM_CLASS = len(cfg.CLASSES)

	# dict from class name to id
	cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(cfg.NUM_CLASS)))

	# x, y, z, intensity, depth
	cfg.INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
	cfg.INPUT_STD = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

	
	# channel number
	cfg.NUM_CHANNEL = 5
	
	# wether to load pre-trained model
	cfg.LOAD_PRETRAINED_MODEL = True
	
	# path to load the pre-trained model
	cfg.PRETRAINED_MODEL_PATH = ''
	
	

	return cfg
