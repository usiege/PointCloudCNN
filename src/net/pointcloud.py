import tensorflow as tf
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import sys
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from cnn import *
import tf_util as tfu


class PointCloud(CNN):
    """docstring for PointCloud"""
    
    def __init__(self, arg):
        super(PointCloud, self).__init__(arg)
        self._model = arg
    
    @property
    def is_training(self):
        return self._is_training
    
    @property
    def model(self):
        return self._model


    def get_input_tensor(self):
        mc = self.model.mc
        flags = self.model.FLAGS
        
        batch_size = flags.batch_size
        num_point = flags.num_point
        num_channnel = 5
        
        pointclouds_pl = tf.placeholder(tf.float32,
                                        shape=(batch_size, num_point, num_channnel))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
        return pointclouds_pl, labels_pl
    
    def get_is_training_tensor(self):
        return tf.placeholder(tf.bool, shape=())
    
    def get_loss(self, pred, label):
        """ pred: B,N,C
            label: B,N """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        return tf.reduce_mean(loss)

    # take input tensor to network
    def get_net(self, point_cloud, is_training):
        self._is_training = is_training
        return self._pointnet_module(point_cloud)

    
    def _placeholder_inputs(self):
        return None
    

	# pointnet base network    
    def _pointnet_module(self, point_cloud, bn_decay=None):
        """ ConvNet baseline, input is BxNxC (batch, number, channels)"""
        print("pointnet cnn ...\n")
        md = self.model
        mc = md.mc
        
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        channel_size = point_cloud.get_shape()[2].value
        
        is_training = self.is_training
        
        net = None
        input_image = tf.expand_dims(point_cloud, -1)  # B*N*C*1
        # conv
        net = tfu.conv2d(input_image, 64, [1, channel_size], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
        net = tfu.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
        net = tfu.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
        net = tfu.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
        points_feat1 = tfu.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                                      bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
        
        # MAX
        pc_feat1 = tfu.max_pool2d(points_feat1, [num_point, 1], padding='VALID', scope='maxpool1')
        
        # FC
        pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
        pc_feat1 = tfu.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        pc_feat1 = tfu.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        print(pc_feat1)
        
        # CONCAT
        pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
        points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])
        
        # CONV
        net = tfu.conv2d(points_feat1_concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='conv6')
        net = tfu.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='conv7')
        net = tfu.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        
        net = tfu.conv2d(net, mc.NUM_CLASS, [1, 1], padding='VALID', stride=[1, 1],
                             activation_fn=None, scope='conv8')
        net = tf.squeeze(net, [2])
        
        return net
    
    

