#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-10-11 下午8:28
# software:PyCharm

import component as ct
import numpy as np
import os

NUM_CLASS = 4

# lidar2d data
# compute lidar2d data class point number




# alibaba data
def static_data(dir):
    
    tool = ct.InputData()
    names = tool.load_subnames(dir)
    
    total_weight = [0] * NUM_CLASS
    for file_name in names:
        file_path = os.path.join(dir, file_name)
        
        ndata = np.reshape(np.load(file_path), (-1, 6))
        
        def single_np(arr, target):
            arr = np.array(arr)
            mask = (arr == target)
            arr_new = arr[mask]
            return arr_new.size
        
        # statistic class weight
        weight = [0] * NUM_CLASS
        for l in range(NUM_CLASS):
            weight[l] = single_np(ndata[:, 5], l)
        
        for i in range(NUM_CLASS):
            total_weight[i] += weight[i]

        if False:
            print("文件：" + file_name)
            print(weight)
            print("")
    
    print(total_weight)
    
        
        
        

if __name__ == "__main__":
    
    print("统计生成文件数据：")

    lidar_path = "/home/mengweiliang/lzh/SqueezeSeg/data/lidar_2d"
    npy_path = "/home/mengweiliang/lzh/SqueezeSeg/data/npy"
    npy_180 = "/home/mengweiliang/lzh/SqueezeSeg/data/npy180"
    npy_360 = "/home/mengweiliang/lzh/SqueezeSeg/data/npy360"
    
    static_data(lidar_path)
    
    
    