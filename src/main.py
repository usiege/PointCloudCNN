#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:19-3-20 下午4:51
# software:PyCharm

import os
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=True))
        print(sess.run(c))