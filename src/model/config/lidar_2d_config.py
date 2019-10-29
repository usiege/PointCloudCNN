import numpy as np
from config import *


def squeezeseg_config():
    
    sc = base_model_config('squeezeseg')

    # image height
    sc.IMAGE_HEIGHT = 64

    # image width
    sc.IMAGE_WIDTH = 512
    
    # weight decay
    sc.WEIGHT_DECAY = 0.0001

    # small value used in batch normalization to prevent dividing by 0. The
    # default value here is the same with caffe's default value.
    sc.BATCH_NORM_EPSILON = 1e-5

    # small value used in denominator to prevent division by 0
    sc.DENOM_EPSILON = 1e-12

    # number iter
    sc.RCRF_ITER = 3
    
    # bi_filter_coef
    sc.BI_FILTER_COEF = 0.1
    
    # ang_filter_coef
    sc.ANG_FILTER_COEF = 0.02
    
    # theta
    sc.ANG_THETA_A = np.array([.9, .9, .6, .6])
    sc.BILTERAL_THETA_A = np.array([.9, .9, .6, .6])
    sc.BILTERAL_THETA_R = np.array([.015, .015, .01, .01])

    return sc


def lidar_2d():
    
    mc = base_model_config('lidar_2d')

    # for squeezeseg
    mc.SQUEEZESEG = squeezeseg_config()

    # data root path
    mc.DATA_ROOT_DIR = "/home/charles/dataset"
    
    # data path
    mc.DATA_PATH = '/home/charles/dataset/squeezeseg/lidar_2d'
    
    # imageset path
    mc.IMAGESET_PATH = '/home/charles/dataset/squeezeseg/ImageSet'
    
    # default log path
    mc.LOG_DIR = './log'
    
    # pre-trained model path
    mc.PRETRAINED_MODEL_PATH = '/home/charles/dataset/squeezeseg/squeezenet'
    
    
    return mc
