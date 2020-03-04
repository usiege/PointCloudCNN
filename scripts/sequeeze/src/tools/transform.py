#!/usr/bin/python
# --encoding=utf8--

import component as ct
import pandas as pd
import numpy as np

import os
import time

from component import InputData

def save_names_file(rootpath, save_file):
    
    tools = ct.InputData()
    names = tools.load_subnames(rootpath)
    
    with open(save_file, 'w') as f:
        for name in names:
            context = name + '\n'
            f.write(context)
        f.close()

# transfortesting data
def transform_test_data(rootpath=""):
    
    print "test root path is {0}".format(rootpath)
    
    tools = ct.InputData(rootpath)
    tools.savePath = 'npy/'

    test_name_path = rootpath + "/intensity_2"
    test_file_names = tools.load_subnames(test_name_path)
    
    for index, file_name in enumerate(test_file_names):
        
        if file_name[-4:] == '.csv':
            npypath = tools.savePath + file_name[:-4] + '.npy'
            
            if os.path.exists(npypath):
                continue
            
            data = tools.cover_csv_to_np(file_name,
                                         intensitydirname='intensity_2', ptsdirname='pts_2')
            # formatdata = tools.generate_image_np(data, debug=False)
            result = data.values
            # print np.shape(result)
            np.save(npypath, result)
            
            # if np.shape(formatdata) == (64, 512, 6):
            #     print '%s has generated' % (npypath)
            

# transform training data
def transform_training_npy(rootpath="", angle=90, debug=False):
    
    print rootpath
    
    trantool = ct.InputData(rootpath)
    # trantool.rootPath = rootpath
    
    if angle == 90:
        trantool.savePath = "npy"
    elif angle == 180:
        trantool.savePath = "npy180"
    elif angle == 360:
        trantool.savePath = "npy360_full"
        
    elif angle == -1:
        trantool.savePath = 'npy_whole'
    else:
        trantool.savePath = "npy_origin"
        
    print('save path is :' + trantool.savePath)
    
    # ptsfiles = trantool.load_file_names()

    filesname_savepath = "../../scripts/log/filenames.txt"
    # read filenames from filenames.txt
    ptsfiles = []
    with open(filesname_savepath, 'r') as f:
        line = f.readline()
        while line:
            line = line[:-1]
            ptsfiles.append(line)
            line = f.readline()
        f.close()
    
    # store filenames into filenames.txt
    # with open(filesname_savepath, 'w') as f:
    #     for i in range(0, len(ptsfiles)):
    #         context = ptsfiles[i] + '\n'
    #         f.write(context)
    #     f.close()
    
    
    idx = 0
    NUM_CLASS = 8
    
    for file in ptsfiles:
        # print '正在转换 file ：%s  ......' % file
        idx += 1
        
        if file[-4:] == '.csv':
            prename = 'channelVELO_TOP_0000_%05d' % idx
            npyname = (prename + '.npy')

            npypath = os.path.join(trantool.savePath, npyname)
            if os.path.exists(npypath):
                continue
            
            if idx == 32310:
                print(idx)
            
            data = trantool.cover_csv_to_np(file, savecsv=False)
            
            # start = time.time()
            if angle == 90 or angle == 180:
                formatdata = trantool.generate_image_np(data, angle=angle, debug=False)
            elif angle == 360:
                # 32310
                formatdata = trantool.generate_image_np360(data.values)
                
            elif angle == -1:
                formatdata = np.reshape(data, (-1, 6))
                if idx == 1:
                    print(np.shape(formatdata))
            else:
                formatdata = np.reshape(data.values[:32768, :6], (64, 512, 6))
                if idx == 1:
                    print(np.shape(formatdata))

            # npy store
            if True:
                np.save(npypath, formatdata)
                
                
            # csv store
            if False:
                if angle == 90:
                    cvs_trail = "csv"
                elif angle == 180:
                    cvs_trail = "image_csv"
                elif angle == 360:
                    cvs_trail = "image_csv360"
                    
                csvpath = os.path.join("./", cvs_trail, file)
                
                if not os.path.exists(csvpath):
                    print("生成文件检查：")
                    print(csvpath)
                    pddata = pd.DataFrame(np.reshape(formatdata, (32768, 6)), \
                                          columns=['x', 'y', 'z', 'intensity', 'range', 'category'])
                    savedata = pddata[['x', 'y', 'z', 'intensity', 'range', 'category']].astype('float32')
                    savedata.to_csv(csvpath)
            
            
            if debug:
                if np.shape(formatdata) == (64, 512, 6):
                    print '%s 已生成' % npypath


    
                
        
if __name__ == '__main__':

    path = "/home/mengweiliang/disk15/df314/training"
    transform_training_npy(path, angle=360)

    # cluster 8 to 4 class
    # npy_cluster = "/home/mengweiliang/disk15/df314/training/npy_cluster360"
    # transform_8_to_4_npy(npy_cluster, angle=360)
    
    # testpath = "/home/mengweiliang/lzh/SqueezeSeg/data/test2"
    # # testpath = '/home/mengweiliang/disk15/df314/test'
    #
    # test_name_path = testpath + "/intensity_2"
    # save_path = "../../scripts/testnames.txt"
    #
    # # save test file names
    # if not os.path.exists(save_path):
    #     save_names_file(test_name_path, save_path)
    #
    #
    # transform_test_data(testpath)