import os
import numpy as np

from easydict import EasyDict as edict


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
