#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-10-12 上午11:06
# software:PyCharm

# from component
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


# 转换成需要的npy格式 for循环写法
def generate_np_format(self, source, statistic=False):
    n = 0  #
    max_phi, min_phi = 0, float('inf')
    
    cCount = [0] * 8
    updateCount = [0] * 2
    
    # height x width x {x, y, z, intensity, range, label}
    npy = np.zeros((64, 512, 6), dtype=np.float16)
    
    start = time.time()
    for indexs in source.index:
        
        # 取出列表每行的值
        values = source.loc[indexs].values[:]
        x, y, z, i, r, c = values[0], values[1], values[2], values[3], values[4], values[5]
        
        if x < 0.5: continue  # 前向
        if abs(y) < 0.5: continue
        
        # theta -16~16 phi 45~135
        theta, phi = self.get_degree(x, y, z)
        
        # 由x, y, z计算出的点
        ptx, pty = self.get_point(theta, phi)
        
        if not self.isempty(npy[ptx, pty, 0:3]):  # 该点上已经有值
            
            lastpoint = npy[ptx, pty, :]
            
            if lastpoint[5] == 0:  # 0表示不关心的点 category == 0
                npy[ptx, pty, :] = [x, y, z, i, r, c]
                updateCount[0] += 1
            
            elif r < lastpoint[4]:
                npy[ptx, pty, :] = [x, y, z, i, r, c]
                updateCount[1] += 1
        
        
        else:
            npy[ptx, pty, :] = [x, y, z, i, r, c]
        
        if statistic:
            if n == 0:
                print ptx, pty
                print 'values test: '
                print type(values)
                print (values)
                print (x, y, z)
                print theta, phi
                n += 1
            
            # 结果统计
            if phi > max_phi: max_phi = phi
            if phi < min_phi: min_phi = phi
            if c < 8: cCount[int(c)] += 1
    end = time.time()
    
    if statistic:
        # print 'point count is: %d' % (self._array_flag_count(xyflag, 1))
        print 'duration is %s' % (end - start)
        print 'phi max and min: %f %f' % (max_phi, min_phi)
        print 'category count statistic: %s' % cCount
        print 'update count statistic: %s' % updateCount
    
    return npy


import os, zipfile

def zip_answers(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    zipf.close()


if __name__ == '__main__':
    source_dir = '/home/mengweiliang/lzh/SqueezeSeg/data/alibaba'
    output_filename = 'answers.zip'
    zip_answers(source_dir, output_filename)