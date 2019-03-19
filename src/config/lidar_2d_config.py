import numpy as np
from config import *


def lidar_2d_config():
    
    mc = base_model_config('lidar_2d')
    
    
    # data path
    mc.DATA_PATH = '/home/charles/dataset/squeezeseg/lidar_2d'
    
    # imageset path
    mc.IMAGESET_PATH = '/home/charles/dataset/squeezeseg/ImageSet'
    
    # default log path
    mc.LOG_DIR = './log'
    
    
    return mc
