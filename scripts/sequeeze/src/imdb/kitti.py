# Author: Bichen Wu (bichen@berkeley.edu) 02/27/2017
# --encoding=utf8-
"""Image data base class for kitti"""

import os
import numpy as np
import subprocess

from imdb import imdb
from config.project_config import *

class kitti(imdb):
    
    def __init__(self, image_set, data_path, mc):
        imdb.__init__(self, 'kitti_'+image_set, mc)
        self._image_set = image_set
        self._data_root_path = data_path
        
        # self._lidar_2d_path = os.path.join(self._data_root_path, 'lidar_2d')
        # self._gta_2d_path = os.path.join(self._data_root_path, 'gta')
        
        self._work_dir = DATA_WORKING_PATH
        
        # lidar_2d path with npy
        self._lidar_2d_path = os.path.join(self._data_root_path, 'lidar_2d')
        
        
        # alibaba data path
        self._ali_path = os.path.join(FLAGS.data_path, self._work_dir)
        

        # a list of string indices of images in the directory
        # self._image_idx = self._load_new_image_set_idx()
        # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
        # the image width and height
        
        ## batch reader ##
        self._perm_idx = None
        self._cur_idx = 0
        # TODO(bichen): add a random seed as parameter
        self._shuffle_image_idx()


    # 新路径, 面向alibaba数据的
    def _lidar_2d_new_path_at(self, idx, angle=90):
        
        # npy_cluster deprecated
        #
        #
        if self._work_dir == 'npy_origin' or \
            self._work_dir == 'npy_whole' or \
            self._work_dir == 'npy' or \
            self._work_dir == 'npy180' or \
            self._work_dir == 'npy360' or \
            self._work_dir == 'npy360_full':
            
            path =  self._ali_path + '/channelVELO_TOP_0000_%05d.npy' % (int(idx))
        else:
            
            path = self._ali_path + '/channelVELO_TOP_0000_%05d-%d.npy' % (int(idx), int(angle))
            
        assert os.path.exists(path), 'File does not exist: {}'.format(path)
        
        return path



    ###
    def _load_new_image_set_idx(self):
        # path_pre = 'channelVELO_TOP_0000_'
        idxs = [i for i in range(self.train_count + 1, self.total_count + 1)]

        return idxs
    
    
    
    
        
        
        
        
        
        