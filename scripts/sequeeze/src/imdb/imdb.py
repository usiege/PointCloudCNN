# Author: Bichen Wu (bichen@berkeley.edu) 02/27/2017
# --encoding=utf8-

"""The data base wrapper class"""

import os
import random
import shutil

import numpy as np
import tensorflow as tf

from utils.util import *
from tools.transfer import *
from config.project_config import *
from nodes import loader

class imdb(object):
    """Image database."""
    
    start_index = 0
    total_count = TRAINING_TOTAL_COUNT
    train_count = TRAINING_TOTAL_COUNT - EVALUATION_TOTAL_COUNT
    
    def __init__(self, name, mc):
        self._name = name
        self._image_set = []
        self._image_idx = []
        self._data_root_path = []
        self.mc = mc
        
        # batch reader
        self._perm_idx = []
        
        self._index_list = [i+1 for i in np.arange(self.train_count)]
        self._image_idx = [i+1 for i in range(self.train_count, self.total_count)]
        
        self._cur_idx = 0
    
    @property
    def name(self):
        return self._name
    
    @property
    def image_idx(self):
        return self._image_idx
    
    @property
    def image_set(self):
        return self._image_set
    
    @property
    def data_root_path(self):
        return self._data_root_path
    
    def _shuffle_image_idx(self):
        self._perm_idx = [self._image_idx[i] for i in
                          np.random.permutation(np.arange(len(self._image_idx)))]
        self._cur_idx = 1
    
    
    def read_batch_new(self, shuffle=True):
        
        mc = self.mc
        
        if shuffle:
            if self._cur_idx + mc.BATCH_SIZE >= self.train_count:
                self._index_list = [self._index_list[i] for i in np.random.permutation(np.arange(len(self._index_list)))]
                self._cur_idx = 1
                
            batch_idx = self._index_list[self._cur_idx: self._cur_idx + mc.BATCH_SIZE]
            self._cur_idx += mc.BATCH_SIZE
        else:
            if self._cur_idx + mc.BATCH_SIZE >= self.total_count-self.train_count:
                self._image_idx = [self._image_idx[i] for i in np.random.permutation(np.arange(self.total_count-self.train_count))]
                self._cur_idx = 1

            batch_idx = self._image_idx[self._cur_idx: self._cur_idx + mc.BATCH_SIZE]
            self._cur_idx += mc.BATCH_SIZE
            
        
        # lidar input: batch * height * width * 5
        lidar_per_batch = []
        # lidar mask, 0 for missing data and 1 otherwise: batch * height * width * 1
        lidar_mask_per_batch = []
        # point-wise labels : batch * height * width
        label_per_batch = []
        # loss weights for different classes: batch * height * width 
        weight_per_batch = []

        for idx in batch_idx:
            
            # read ali batch data
            record = np.load(self._lidar_2d_new_path_at(idx, angle=360))\
                .astype(np.float32, copy=False)
            
            # # transform data
            # ld = loader()
            # record = ld.pto_depth_map(record)
            # record = trainging_data(record)
            
            # [::-1] ----> [-1:-len()-1:-1] reverse
            if mc.DATA_AUGMENTATION:
                if mc.RANDOM_FLIPPING: # very bad
                    if np.random.rand() > 0.5:
                        # reverse y
                        record = record[:, ::-1, :]
                        record[:, :, 1] *= -1

            lidar = record[:, :, :5] # x, y, z, intensity, range
            # zenith_level 64 azimuth_level 512
            lidar_mask = np.reshape((lidar[:, :, 4] > 0), \
                                    [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1])

            #normalize
            lidar = (lidar - mc.INPUT_MEAN) / mc.INPUT_STD

            label = record[:, :, 5]
            weight = np.zeros(label.shape)
            
            # each label 's weight
            for l in range(mc.NUM_CLASS):
                # print mc.CLS_LOSS_WEIGHT[int(l)]
                weight[label==l] = mc.CLS_LOSS_WEIGHT[int(l)]

            lidar_per_batch.append(lidar)
            lidar_mask_per_batch.append(lidar_mask)
            label_per_batch.append(label)
            weight_per_batch.append(weight)

        return np.array(lidar_per_batch), np.array(lidar_mask_per_batch), \
               np.array(label_per_batch), np.array(weight_per_batch)
    
    
    def evaluate_detections(self):
        raise NotImplementedError
