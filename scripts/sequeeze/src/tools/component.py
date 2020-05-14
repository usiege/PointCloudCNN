#!/usr/bin/python3
# --encoding=utf8--

import os
import numpy as np
import pandas as pd
import cmath, math
import time

class OutputData(object):
    
    @property
    def modelPath(self):
        return self._modelPath
    
    @modelPath.setter
    def modelPath(self, path):
        self._modelPath = path


        
ANGLE_PHI_MAX = 360.0
ANGLE_PHI_MIN = 0.0

ANGLE_THETA_MAX = 16.0
ANGLE_THETA_MIN = -16.0

class InputData(object):
    
    # 数据根目录
    @property
    def rootPath(self):
        return self._rootPath
    
    @rootPath.setter
    def rootPath(self, path):
        self._rootPath = path
        
    # 生成的npy数据保存目录
    @property
    def savePath(self):
        return self._savePath
    @savePath.setter
    def savePath(self, trail_path):
        self._savePath = os.path.join(self.rootPath, trail_path)

    
    def __init__(self, path=''):
        self.rootPath = path
        pass

    # 加载所有的文件名
    def load_file_names(self, subpath="pts"):
        assert self.rootPath != "", "root path is empty"
        rootname = self.rootPath + '/' + subpath

        return self.load_subnames(rootname)
    
    
    def load_subnames(self, rootpath):
        
        result = []
        ext = ['csv', 'npy']
    
        files = self._filenames(rootpath)
        for file in files:
            if file.endswith(tuple(ext)):
                result.append(file)
    
        return result

    # 所有子文件
    def _filenames(self, filedir):
        result = []
        for root, dirs, files in os.walk(filedir):
            # print "root: {0}".format(root)
            # print "dirs: {0}".format(dirs)
            # print "files: {0}".format(files)
            result = files
        return result

    # 统计标记数量
    def _array_flag_count(self, array, flag):
        count = 0
        for num in array:
            if num == flag:
                count += 1
        return count

    
    
    def get_degree(self, x, y, z):
    
        sqrt_xy = np.sqrt(x ** 2 + y ** 2)
        # sqrt_xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
        theta = np.arctan(z / sqrt_xy) * 180 / math.pi
    
        # phi
        phi = np.arcsin(y / sqrt_xy) * 180 / math.pi
    
        # print(theta, phi)
    
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

    def get_point(self, theta, phi):
    
        # image x(height) * y(width) 2d
        # 向下取整
        x = int((theta - ANGLE_THETA_MIN) / ((ANGLE_THETA_MAX - ANGLE_THETA_MIN) / 64))
        y = int((phi - ANGLE_PHI_MIN) / ((ANGLE_PHI_MAX - ANGLE_PHI_MIN) / 512))
    
        # 严防越界
        x = (x > 63) and 63 or x
        y = (y > 511) and 511 or y
    
        return x, y

    def get_thetaphi(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
    
        return self.get_point(theta, phi)

    def get_point_theta(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
    
        return self.get_point(theta, phi)[0]

    def get_point_phi(self, x, y, z):
        theta, phi = self.get_degree(x, y, z)
    
        return self.get_point(theta, phi)[1]

    def isempty(self, x):
        if (x == [0, 0, 0]).all():
            return True
        else:
            return False
        
    

    # 将csv转成npy文件
    # training下的三个目录中的文件名相同,分别提取出来合并成一个cvs文件
    # 将csv文件转换成与Seg相似的NPY文件
    
    # seg中npy文件格式为[x, y, z, intensity, range, category], 共有 64 * 512 = 32768个
    # 而我们的每个csv有57888个点。
    
    # 返回一个(57888*, 6)
    def cover_csv_to_np(self, filename, savecsv=False,
                            ptsdirname='pts',
                            intensitydirname='intensity',
                            categorydirname='category'):
        # print filename
        
        rootpath = self.rootPath
        ptsPath = self.rootPath + '/' + ptsdirname + '/' + filename
        
        intensityPath = os.path.join(self.rootPath, intensitydirname,filename)
        # self.rootPath + '/' + intensitydirname + '/' + filename
        categoryPath = os.path.join(self.rootPath, categorydirname, filename)
        # categoryPath = self.rootPath + '/' + categorydirname + '/' + filename
        
        # npypath = os.path.join(self.savePath, filename)
        # npypath = self.savePath + '/' + filename

        # pts.columns = ['x', 'y', 'z', 'i', 'c']
        pts = pd.read_csv(ptsPath, header=None)
        intensity = pd.read_csv(intensityPath, header=None)
        
        if os.path.exists(categoryPath):
            category = pd.read_csv(categoryPath, header=None) # dtypes
        else:
            category = pd.DataFrame(np.zeros(np.shape(intensity), np.float32))
        
        # print '---- pts ----'
        # print pts.ix[1]
        
        # 连接操作
        contact = pd.concat([pts, intensity, category], axis=1)
        #
        data = pd.DataFrame(contact)
        data.columns = ['x', 'y', 'z', 'i', 'c']
        data.insert(4, 'r', 0)
        
        
        # print '----- contact -----'
        data['r'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2)
        
        # print data
        # nzero = data.loc[(data.x != 0) & (data.y != 0) & (data.z != 0)]
        
        # print nzero
        if savecsv:
            csv_path = os.path.join("./", "csv", filename)
            print("csv file : %s" % csv_path)
            if not os.path.exists(csv_path):
                data.to_csv(csv_path, index=False, header=False)
    
        return data
    
    # testing transform npy
    # transfer.py
        
        
    # 转换成npy格式 np
    def generate_image_np(self, source, angle=90, debug=False):
        
        # print(type(source))
        if type(source) == np.ndarray:
            data = source
        else:
            data = source.values
        
        if angle == 90:
            ANGLE_PHI_MAX = 135
            ANGLE_PHI_MIN = 45
        elif angle == 180:
            ANGLE_PHI_MIN = 0
            ANGLE_PHI_MAX = 180
        
        # print type(data)
        
        x = [data[i][0] for i in range(len(data[:, 0]))]
        y = [data[i][1] for i in range(len(data[:, 0]))]
        z = [data[i][2] for i in range(len(data[:, 0]))]
        
        intensity = [data[i][3] for i in range(len(data[:, 0]))]
        distance = [data[i][4] for i in range(len(data[:, 0]))]
        label = [data[i][5] for i in range(len(data[:, 0]))]
        

        thetaPt = [self.get_point_theta(data[i][0], data[i][1], data[i][2]) for i in range(len(data[:, 0]))]  # x
        phiPt = [self.get_point_phi(data[i][0], data[i][1], data[i][2]) for i in range(len(data[:, 0]))]  # y
        
        # 生成数据 phi * theta * [x, y, z, i, r, c]
        image = np.zeros((64, 512, 6), dtype=np.float16)

        def store_image(index):
            # print (theta[index], phi[index])
   
            image[thetaPt[index], phiPt[index], 0:3] = [x[index], y[index], z[index]]
            image[thetaPt[index], phiPt[index], 3] = intensity[index]
            image[thetaPt[index], phiPt[index], 4] = distance[index]
            image[thetaPt[index], phiPt[index], 5] = label[index]
        
        for i in range(len(x)):
            if x[i] < 0.5: continue # 前向
            if abs(y[i]) < 0.5: continue
            
            if self.isempty(image[thetaPt[i], phiPt[i], 0:3]):
                store_image(i)
            elif label[i] == image[thetaPt[i], phiPt[i], 5]:
                if distance[i] < image[thetaPt[i], phiPt[i], 4]:
                    image[thetaPt[i], phiPt[i], 4] = distance[i]
            elif image[thetaPt[i], phiPt[i], 5] == 0 and label[i] != 0:
                store_image(i)
            else:
                if distance[i] < image[thetaPt[i], phiPt[i], 4]:
                    store_image(i)
        
        
        return image


    # 将所有数据转换成网格需要的格式
    def generate_image_np360(self, values):

        data = values

        # data collection
        x = [data[i][0] for i in range(len(data[:, 0]))]
        y = [data[i][1] for i in range(len(data[:, 0]))]
        z = [data[i][2] for i in range(len(data[:, 0]))]

        intensity = [data[i][3] for i in range(len(data[:, 0]))]
        distance = [data[i][4] for i in range(len(data[:, 0]))]
        label = [data[i][5] for i in range(len(data[:, 0]))]

        def get_image_point(x, y, z, distance):
    
            sqrt_xy = np.sqrt(x ** 2 + y ** 2)
            
            theta = np.arctan(z / sqrt_xy) * 180 / math.pi
            phi = np.arcsin(y / sqrt_xy) * 180 / math.pi
    
            # print(x, y)
            # print("(%s) ----------> " % phi)
            
            if theta > ANGLE_THETA_MAX:
                theta = ANGLE_THETA_MAX
            if theta < ANGLE_THETA_MIN:
                theta = ANGLE_THETA_MIN
    
            if x > 0 and y > 0:  # 0 - 90
                phi = phi
            elif x < 0 and y > 0:  # 90 -180
                phi = 180 - phi
            elif x < 0 and y < 0:  # 180 - 270
                phi = 180 - phi
            elif x > 0 and y < 0:  # 270 - 360
                phi = phi + 360
            
    
            # print("phi is : (%s)" % phi)
            # print()
    
            theta_pt = int((theta - (ANGLE_THETA_MIN)) / ((ANGLE_THETA_MAX-ANGLE_THETA_MIN) / 64))
            phi_pt = int((phi - ANGLE_PHI_MIN) / ((ANGLE_PHI_MAX-ANGLE_PHI_MIN) / 512))
    
            # 严防越界
            theta_pt = (theta_pt > 63) and 63 or theta_pt
            phi_pt = (phi_pt > 511) and 511 or phi_pt
    
            return theta_pt, phi_pt

        # image with theta axis and phi axis
        thetaPt = [get_image_point(x[i], y[i], z[i], distance[i])[0] \
                   for i in range(len(data[:, 0]))]  #
        phiPt = [get_image_point(x[i], y[i], z[i], distance[i])[1] \
                 for i in range(len(data[:, 0]))]

        # generate image
        image = np.zeros((64, 512, 6), dtype=np.float16)

        def store_image(index):
            # print (theta[index], phi[index])
            
            xp = int(thetaPt[index])
            yp = int(phiPt[index])
            
            image[xp, yp, 0:3] = [x[index], y[index], z[index]]

            image[xp, yp, 3] = intensity[index]
            image[xp, yp, 4] = distance[index]
            image[xp, yp, 5] = label[index]
            

        ignore_accuracy = 0.005
        for i in range(len(x)):
            
            if abs(x[i]) < ignore_accuracy: continue
            if abs(y[i]) < ignore_accuracy: continue
            
            xyz = image[thetaPt[i], phiPt[i], 0:3]
            
            l = image[thetaPt[i], phiPt[i], 5]
            d = image[thetaPt[i], phiPt[i], 4]
            
            if self.isempty(xyz):
                store_image(i)
            
            # 点上是标签0的被替换
            elif l == 0 and label[i] != 0:
                store_image(i)

            # 同一个标签近的值
            elif label[i] == l and distance[i] < d:
                store_image(i)
                
            else:
                store_image(i)
                
        return image
    
        
if __name__ == '__main__':
    
    testpath = '../../data/training'
    
    compontent = TransformData()
    compontent.rootPath = testpath
    # print(compontent.load_file_names(testpath))

    print '正在转换......'
    result = compontent.cover_csv_to_nzero('ac3fc22d-f288-477f-a7aa-b73936b23f91_channelVELO_TOP.csv')
    print type(result)

    slow = True
    global formatdata
    
    if not slow:
        formatdata = compontent.generate_np_format(result, statistic=True)
    else:
        formatdata = compontent.generate_image_np(result, debug=True)

    print '转换后数据：'
    # 生成一个csv文件
    pdata = pd.DataFrame(np.reshape(formatdata, (-1, 6)), columns=['x', 'y', 'z', 'intensity', 'range', 'category'])
    pdata[['x', 'y', 'z', 'intensity', 'range', 'category']].astype('float64').to_csv('transnpy_quick.csv', index=None,
                                                                                      header=None)
    # 生成一个npy文件
    np.save("./data_quick.npy", formatdata)

    # 检查npy文件
    npy = np.load("./data_quick.npy")

    print '文件转换已完成！'
    print np.shape(npy)
    

    
