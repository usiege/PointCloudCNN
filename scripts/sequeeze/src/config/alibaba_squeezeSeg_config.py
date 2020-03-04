
import numpy as np

from config import base_model_config

def alibaba_squeezeSeg_config():
    
    asc = base_model_config('KITTI')
    
    asc.CLASSES = [
        'DontCare',
        'cyclist',
        'tricycle',
        'sm_allMot',
        'bigMot',
        'pedestrian',
        'crowds',
        'unknown',
    ]
    # asc.CLASSES = ['unknown', 'car', 'pedestrian', 'cyclist']
  
    asc.NUM_CLASS = len(asc.CLASSES)
    asc.CLS_2_ID = dict(zip(asc.CLASSES, range(len(asc.CLASSES))))
    
    asc.CLS_LOSS_WEIGHT = np.array([1/142.0, 1.0/0.15,  1/0.07, 1.0/3,
                                    1, 8, 1/4.6, 50])
    #
    # # asc.CLS_LOSS_WEIGHT = np.array([1000.0, 1.0, 0.5, 0.02,
    # #                                 0.006, 0.0014, 1.0/6, 0.5])
    #
    asc.CLS_COLOR_MAP = np.array([[1.0, 1.0, 1.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0],
                                  [1.0, 1.0, 0.0],
                                  [1.0, 0.0, 1.0],
                                  [0.0, 1.0, 1.0],
                                  [1.0, 0.5, 0.0]])

    # asc.CLS_LOSS_WEIGHT = np.array([1 / 15.0, 1.0, 10.0, 10.0])
    # asc.CLS_COLOR_MAP = np.array([[0.00, 0.00, 0.00],
    #                              [0.12, 0.56, 0.37],
    #                              [0.66, 0.55, 0.71],
    #                              [0.58, 0.72, 0.88]])

    asc.BATCH_SIZE = 32
    asc.AZIMUTH_LEVEL = 512
    asc.ZENITH_LEVEL = 64

    asc.LCN_HEIGHT = 3
    asc.LCN_WIDTH = 5
    asc.RCRF_ITER = 3

    # asc.BILATERAL_THETA_A = np.array([.9, .9, .6, .6])
    # asc.BILATERAL_THETA_R = np.array([.015, .015, .01, .01])
    # asc.BI_FILTER_COEF = 0.1
    # asc.ANG_THETA_A = np.array([.9, .9, .6, .6])
    
    # bi_angular_filters
    # angular_filter_kernel
    asc.BILATERAL_THETA_A = np.array([.9, .9, .9, .9,
                                      .6, .6, .6, .6])
    asc.ANG_THETA_A = np.array([.9, .9, .9, .9,
                                .6, .6, .6, .6])

    # angular_filters
    asc.BILATERAL_THETA_R = np.array([.015, .015, .015, .015,
                                      .01, .01, .01, .01])

    asc.BI_FILTER_COEF = 0.1
    asc.ANG_FILTER_COEF = 0.02

    asc.CLS_LOSS_COEF = 15.0
    asc.WEIGHT_DECAY = 0.0001 # 0.0001
    asc.LEARNING_RATE = 0.01 # origin 0.01
    asc.DECAY_STEPS = 10000
    asc.MAX_GRAD_NORM = 1.0
    asc.MOMENTUM = 0.9
    asc.LR_DECAY_FACTOR = 0.5

    asc.DATA_AUGMENTATION = False
    asc.RANDOM_FLIPPING = False

    # x, y, z, intensity, distance
    asc.INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
    asc.INPUT_STD = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

    return asc