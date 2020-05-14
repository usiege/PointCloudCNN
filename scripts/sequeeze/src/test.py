#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-9-28 下午8:16
# software:PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time
import glob

import numpy as np
from six.moves import xrange
import tensorflow as tf
from PIL import Image

from config import *
from imdb import kitti
from utils.util import *
from nets import *

import pandas as pd
import time, threading


from tools.component import InputData
from tools.transfer import *
from config.project_config import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('gpu', '3', """gpu id.""")


def test():
    """Detect LiDAR data."""
    
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    with tf.Graph().as_default():
        
        mc = alibaba_squeezeSeg_config()
        mc.LOAD_PRETRAINED_MODEL = False
        mc.BATCH_SIZE = 1  # TODO(bichen): fix this hard-coded batch size.
        model = SqueezeSeg(mc)
        
        saver = tf.train.Saver(model.model_params)
        # saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess, TESTING_MODEL_PATH)
            
            # get lidar testing data pred class
            # lidar ---> (64, 512, 5)
            # pred_cls ----> (64, 512, 1)
            def generate_pred_cls(f, mc, model, sess):
                
                if f.shape[2] > 5:
                    lidar = f[:,:, 0:5]
                else:
                    lidar = f
    
                lidar_mask = np.reshape(
                    (lidar[:, :, 4] > 0),
                    [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
                )
                lidar = (lidar - mc.INPUT_MEAN) / mc.INPUT_STD
    
                pred_cls = sess.run(
                    model.pred_cls,
                    feed_dict={
                        model.lidar_input: [lidar],
                        model.keep_prob: 1.0,
                        model.lidar_mask: [lidar_mask]
                    }
                )
                pred_cls = np.reshape(pred_cls, (-1, 1))
                
                return pred_cls
            
            # generate result with pre_cls and
            def npy_to_image_to_result(fnpy, sess):
                
                # transform origin npy to image(64, 512, 6)
                image = testing_data(fnpy)
                
                indexes = np.reshape(image[:, :, 6], (-1, 1))
                
                xp = np.reshape(image[:, :, 7], (-1, 1))
                yp = np.reshape(image[:, :, 8], (-1, 1))
                
                # generate pred results with image npy data
                pred_cls = generate_pred_cls(image[:,:, 0:6], mc, model, sess)
            
                result = np.zeros((fnpy.shape[0], 1))
                # image = np.reshape(image, (-1, 6))
            
                def store_result(i):
                    
                    idx = int(indexes[i, 0])
                    idxp = int(xp[i, 0])
                    idyp = int(yp[i, 0])
                    
                    if idx == 57920:
                        pass
                    
                    result[idx, 0] = pred_cls[i, 0]
                
                count = indexes.shape[0]
                _ = [store_result(i) for i in range(count)]
                
                return result
            
            
            # save data
            def save_result(result, f):
    
                file_name = f.strip('.npy').split('/')[-1]
                file_path = FLAGS.out_dir + file_name + '.csv'
                
                # csv
                cvsdata = pd.DataFrame(result, columns=['category'])
                if not os.path.exists(file_path):
                    cvsdata[['category']].astype('int32').to_csv(file_path, index=None, header=None)
                    
                # npy
                if False:
                    np.save(
                        os.path.join(FLAGS.out_dir, 'pred_' + file_name + '.npy'),
                        pred_cls[0]
                    )
                
                
            def enqueue(f, sess):
                # load file
                fnpy = np.load(f).astype(np.float32, copy=False)
    
                # produce result
                result = npy_to_image_to_result(fnpy, sess)
    
                # restore data (58***, 1)
                # result = transform_label(result)
    
                # save result
                save_result(result, f)
            
            def queue_action(file_list, sess, coord):
                with coord.stop_on_exception():
                    while not coord.should_stop():
                        print("---------&&&&&&\n")
                        for f in file_list:
                            enqueue(f, sess)

            #
            fs = glob.glob(FLAGS.input_path)
            for f in fs:
                # save the data
                file_name = f.strip('.npy').split('/')[-1]
                file_path = FLAGS.out_dir + file_name + '.csv'

                if os.path.exists(file_path):
                    print(file_path)
                    continue

                enqueue(f, sess)


            # coord = tf.train.Coordinator()
            # threads = []
            # NUM_OF_THREAD = 5
            # for t in range(NUM_OF_THREAD):
            #     eqth = threading.Thread(target=queue_action, \
            #                             args=[fs[t::NUM_OF_THREAD], sess, coord])
            #     eqth.start()
            #     threads.append(eqth)
            
            

def main(argv=None):
    
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    
    print('Detection output written to {}'.format(FLAGS.out_dir))
    test()

if __name__ == '__main__':
    tf.app.run()


'''
            # take testing npy to a origin npy
            def origin_npy_result(fnpy):
                
                if np.shape(fnpy)[0] >= 32768:
        
                    f1 = np.load(f).astype(np.float32, copy=False)[:32768, :5]
                    f1 = np.reshape(f1, (64, 512, 5))
        
                    fillnp = np.zeros((32768, 5)).astype(np.float32)
                    f2 = np.load(f).astype(np.float32, copy=False)[32768:, :5]
                    avildable_number = np.shape(f2)[0]
                    padding_number = 32768 - avildable_number  # adding number
                    fillnp[:np.shape(f2)[0], :5] = f2[:]
        
                    # print(np.shape(fnpy))
                    # print(np.shape(f1), np.shape(fillnp))
        
                    fillnp = np.reshape(fillnp, (64, 512, 5))
        
                    pred_cls1 = generate_pred_cls(f1, mc, model, sess)
                    pred_cls2 = generate_pred_cls(fillnp, mc, model, sess)
        
                    result1 = np.reshape(pred_cls1, (32768, 1))
                    result2 = np.reshape(pred_cls2, (32768, 1))
        
                    result = np.zeros((np.shape(fnpy)[0], 1)).astype(np.float32, copy=True)
                    result[:32768, :] = result1
                    result[32768:(32768 + avildable_number), :] = result2[:avildable_number, :]
    
                else:
        
                    f1 = np.zeros((32768, 5))
                    avildable_number = np.shape(fnpy)[0]
                    f1[:np.shape(fnpy)[0], :5] = fnpy[:, :5]
        
                    f1 = np.reshape(f1, (64, 512, 5))
                    pred_cls = generate_pred_cls(f1, mc, model, sess)
        
                    result = np.reshape(pred_cls, (32768, 1))
                    result = result[:avildable_number, :]
                
                return result
'''