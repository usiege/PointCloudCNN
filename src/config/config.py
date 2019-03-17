import os
import numpy as np

from easydict import EasyDict as edict

def base_model_config(dataset=''):

	cfg = edict()

	# imageset files names
	cfg.IMAGESET = ['all', 'test', 'train', 'val']

	# Dataset used to train/val/test model.
	cfg.DATASET = dataset.upper()

	# data path
	cfg.DATA_PATH = ''
	
	# imageset path
	cfg.IMAGESET_PATH = ''
	

	# classes
	cfg.CLASSES = ['unknown', 'car', 'pedestrian', 'cyclist']

	# number of classes
	cfg.NUM_CLASS = len(cfg.CLASSES)

	# dict from class name to id
	cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(cfg.NUM_CLASS)))

	# batch size 
	cfg.BATCH_SIZE = 1

	return cfg
