#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-11-6 下午4:40
# software:PyCharm

import numpy as np
import os

class loader(object):
    
    _angle_theta_min = -16.0
    _angle_theta_max = 16.0
    
    _angle_phi_min = 0.0
    _angle_phi_max = 360.0
    

    def __init__(self):
        pass

    def isempty(self, x):
        if (x == [0, 0, 0]).all():
            return True
        else:
            return False


    def pto_depth_map(self, cloud_points, H=64, W=512, C=6,
                      dtheta=0.5, dphi=360./512.0):
        
        x, y, z, ii = cloud_points[:, 0], cloud_points[:, 1], cloud_points[:, 2], cloud_points[:, 3]
        if C == 6:
            r = cloud_points[:, 4]
            l = cloud_points[:, 5]
        else:
            r = np.sqrt(x ** 2 + y ** 2)
            l = np.zeros(r.shape)
            
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        
        theta = np.rad2deg(np.arctan(z/r))
        phi = np.rad2deg(np.arcsin(y/r))

        phi[(x > 0) & (y > 0)] = phi[(x > 0) & (y > 0)]
        phi[(x > 0) & (y < 0)] = phi[(x > 0) & (y < 0)] + 360.0
        
        phi[(x < 0)] = 180.0 - phi[(x < 0)]
        
        tp = ((theta - self._angle_theta_min) / dtheta).astype(int)
        pp = ((phi - self._angle_phi_min) / dphi).astype(int)
        
        tp[tp<0] = 0
        tp[tp>=64] = 63
        pp[pp<0] = 0
        pp[pp>=512] = 511
        
        # np.deg2rad()
        # np.rad2deg()
        depth_map = np.zeros((H, W, C))

        def store_image(image, index):
            # print (theta[index], phi[index])
    
            xp = int(tp[index])
            yp = int(pp[index])
    
            image[xp, yp, 0:3] = [x[index], y[index], z[index]]
    
            image[xp, yp, 3] = ii[index]
            image[xp, yp, 4] = d[index]
            image[xp, yp, 5] = l[index]

        ignore_accuracy = 0.05
        
        for i in range(len(x)):
    
            if abs(x[i]) < ignore_accuracy: continue
            if abs(y[i]) < ignore_accuracy: continue
    
            xyz = depth_map[tp[i], pp[i], 0:3]
    
            label = depth_map[tp[i], pp[i], 5]
            dis = depth_map[tp[i], pp[i], 4]
    
            if self.isempty(xyz):
                store_image(depth_map, i)
    
            # 点上是标签0的被替换
            elif label == 0 and l[i] != 0:
                store_image(depth_map, i)
    
            # 同一个标签近的值
            elif l[i] == label and d[i] < dis:
                store_image(depth_map, i)
    
            else:
                store_image(depth_map, i)

        
        return  depth_map

if __name__ == '__main__':
    
    path = "/home/mengweiliang/disk15/df314/training/npy_whole"
    file_name = "channelVELO_TOP_0000_50000.npy"
    
    npy = np.load(os.path.join(path, file_name))
    
    ld = loader()
    ld.pto_depth_map(npy)
    