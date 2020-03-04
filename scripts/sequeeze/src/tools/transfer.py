#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-11-5 下午5:23
# software:PyCharm

import numpy as np
import math

ANGLE_PHI_MAX = 360.0
ANGLE_PHI_MIN = 0.0

ANGLE_THETA_MAX = 16.0
ANGLE_THETA_MIN = -16.0

theta_max = 0.0
theta_min = 0.0


def isempty(x):
    if (x == [0, 0, 0]).all():
        return True
    else:
        return False

def get_degree(x, y, z):
    
    sqrt_xy = np.sqrt(x ** 2 + y ** 2)
    # sqrt_xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    theta = np.arctan(z / sqrt_xy) * 180 / math.pi
    
    # phi
    phi = np.arcsin(y / sqrt_xy) * 180 / math.pi
    
    
    # 调整角度
    if y > 0:
        phi = phi
    else:
        phi = phi + 180
    
    # print("degree: %f, %f" % (theta, phi))
    # 防止越界
    if phi > ANGLE_PHI_MAX:
        phi = ANGLE_PHI_MAX
    elif phi < ANGLE_PHI_MIN:
        phi = ANGLE_PHI_MIN
    
    if theta > ANGLE_THETA_MAX:
        theta = ANGLE_THETA_MAX
    elif theta < ANGLE_THETA_MIN:
        theta = ANGLE_THETA_MIN
    
    return theta, phi


def get_point(theta, phi):
    # image x(height) * y(width) 2d
    # 向下取整
    x = int((theta - ANGLE_THETA_MIN) / ((ANGLE_THETA_MAX - ANGLE_THETA_MIN) / 64))
    y = int((phi - ANGLE_PHI_MIN) / ((ANGLE_PHI_MAX - ANGLE_PHI_MIN) / 512))
    
    # 严防越界
    x = (x > 63) and 63 or x
    x = (x < 0) and 0 or x
    y = (y > 511) and 511 or y
    y = (y < 0) and 0 or y

    return x, y

def get_point_theta(x, y, z):
    theta, phi = get_degree(x, y, z)

    return get_point(theta, phi)[0]


def get_point_phi(x, y, z):
    theta, phi = get_degree(x, y, z)
    
    return get_point(theta, phi)[1]


def trainging_data(fnpy):
    
    return _tranfrom_data(fnpy)[:, :, :6]

def testing_data(fnpy):
    
    return _tranfrom_data(fnpy)[:, :, :]

def _pto_depth_map(fnpy):
    
    assert type(fnpy) == np.ndarray, "source is not a ndarray type!!!"
    data = fnpy
    
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    i = data[:, 3]
    # d =
    
    
    

def _tranfrom_data(fnpy):
    
    assert type(fnpy) == np.ndarray, "source is not a ndarray type!!!"
    data = fnpy
    
    data_length = data.shape[0]
    
    x = [data[i][0] for i in range(data_length)]
    y = [data[i][1] for i in range(data_length)]
    z = [data[i][2] for i in range(data_length)]
    
    intensity = [data[i][3] for i in range(data_length)]
    distance = [data[i][4] for i in range(data_length)]
    label = [data[i][5] for i in range(data_length)]
    
    thetaPt = [get_point_theta(data[i][0], data[i][1], data[i][2]) for i in range(data_length)]  # x
    phiPt = [get_point_phi(data[i][0], data[i][1], data[i][2]) for i in range(data_length)]  # y
    
    # 生成数据 phi * theta * [x, y, z, i, r, c]
    # image = np.zeros((64, 512, 6), dtype=np.float16)
    image_index = np.zeros((64, 512, 9), dtype=np.float32)
    
    def store_image(index):
        # print (theta[index], phi[index])
        xp = int(thetaPt[index])
        yp = int(phiPt[index])

        image_index[xp, yp, 0:3] = [x[index], y[index], z[index]]
        
        image_index[xp, yp, 3] = intensity[index]
        image_index[xp, yp, 4] = distance[index]
        image_index[xp, yp, 5] = label[index]
        
        image_index[xp, yp, 6] = index
        image_index[xp, yp, 7] = xp
        image_index[xp, yp, 8] = yp
        
        if xp == 32 and yp == 5:
            pass

        
    for i in range(len(x)):
        
        if i == 57920:
            pass
        
        tp = (thetaPt[i], phiPt[i])
        # print(tp)
        
        if isempty(image_index[tp[0], tp[1], 0:3]):
            store_image(i)
        elif label[i] == image_index[tp[0], tp[1], 5]:
            if distance[i] < image_index[tp[0], tp[1], 4]:
                store_image(i)
        elif image_index[tp[0], tp[1], 5] == 0 and label[i] != 0:
            store_image(i)
        else:
            if distance[i] < image_index[tp[0], tp[1], 4]:
                store_image(i)
    
    
    return image_index