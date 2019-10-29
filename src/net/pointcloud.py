import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import sys
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from cnn import CNN

import tensorflow as tf
import numpy as np

import tf_util as tfu # for pointnet
import squ_util as sul # for squeezeseg
import gn_util as gul # for graph cnn

class PointCloud(CNN):
    """docstring for PointCloud"""
    
    def __init__(self, arg):
        super(PointCloud, self).__init__(arg)
        self._arg = arg
        self._model = None
        
        # model parameters
        self.net_params = []


    @property
    def model(self):
        return self._model
    
    # @model.setter
    # def model(self, arg):
    #     self._model = arg
    
    @property
    def is_training(self):
        return self._is_training


    def get_input_tensor(self, model):
        
        self._model = model
        mc = model.mc
        flags = model.FLAGS
        net = model.FLAGS.net
        
        num_point = flags.num_point
        
        batch_size = flags.batch_size
        image_height = mc.SQUEEZESEG.IMAGE_HEIGHT
        image_width = mc.SQUEEZESEG.IMAGE_WIDTH
        num_channnel = mc.NUM_CHANNEL
        
        if net == 'pointnet':
            pointclouds_pl = tf.placeholder(tf.float32,shape=(batch_size, num_point, num_channnel))
            labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
        elif net == 'squeezeseg':
            pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, image_height,
                                                               image_width, num_channnel))
            labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
        else:
            pointclouds_pl = None
            labels_pl = None
        
        return pointclouds_pl, labels_pl
    
    def get_is_training_tensor(self):
        return tf.placeholder(tf.bool, shape=())
    
    
    def get_loss(self, pred, label):
        """ pred: B,N,C
            label: B,N """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        return tf.reduce_mean(loss)


    # take input tensor to network
    def get_net(self, point_cloud,
                is_training=None, labels=None):
        self.net_params = []
        self._is_training = is_training
        
        return self._pointnet_module(point_cloud)
    
    
    # dynamic graph cnn
    def _dynamic_graph_module(self, point_cloud):
        pass

    
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
    
    
    # pointnet++ base network
    def _pointnet2_net_module(self):
        
        return None

    # squeezeseg series base network
    def _squeezeseg_module(self, point_cloud):
    
        print ('squeezeseg net ...')
        md = self.model
        mc = md.mc
    
        batch_size = point_cloud.get_shape()[0].value
        image_height = point_cloud.get_shape()[1].value
        image_width = point_cloud.get_shape()[2].value
        channel_size = point_cloud.get_shape()[3].value
    
        lidar_input = point_cloud
    
        conv1 = sul.conv_layer('conv1', lidar_input, filters=64, size=3, stride=2,
                               padding='SAME', freeze=False, xavier=True)
        conv1_skip = sul.conv_layer('conv1_skip', lidar_input, filters=64, size=1, stride=1,
                                    padding='SAME', freeze=False, xavier=True)
    
        pool1 = sul.pooling_layer('pool1', conv1, size=3, stride=2, padding='SAME')
    
        fire2 = sul.fire_layer('fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
        fire3 = sul.fire_layer('fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
        pool3 = sul.pooling_layer('pool3', fire3, size=3, stride=2, padding='SAME')
    
        fire4 = sul.fire_layer('fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
        fire5 = sul.fire_layer('fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
        pool5 = sul.pooling_layer('pool5', fire5, size=3, stride=2, padding='SAME')
    
        fire6 = sul.fire_layer('fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
        fire7 = sul.fire_layer('fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
        fire8 = sul.fire_layer('fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
        fire9 = sul.fire_layer('fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    
        # Deconvolation
        fire10 = sul.fire_deconv('fire_deconv10', fire9, s1x1=64, e1x1=128, e3x3=128,
                                 factors=[1, 2], stddev=0.1)
        fire10_fuse = tf.add(fire10, fire5, name='fure10_fuse')
    
        fire11 = sul.fire_deconv('fire_deconv11', fire10_fuse, s1x1=32, e1x1=64, e3x3=64,
                                 factors=[1, 2], stddev=0.1)
        fire11_fuse = tf.add(fire11, fire3, name='fire11_fuse')
    
        fire12 = sul.fire_deconv('fire_deconv12', fire11_fuse, s1x1=16, e1x1=32, e3x3=32,
                                 factors=[1, 2], stddev=0.1)
        fire12_fuse = tf.add(fire12, conv1, name='fire12_fuse')
    
        fire13 = sul.fire_deconv('fire_deconv13', fire12_fuse, s1x1=16, e1x1=32, e3x3=32,
                                 factors=[1, 2], stddev=0.1)
        fire13_fuse = tf.add(fire13, conv1_skip, name='fire13_fuse')
    
        # dropout
        drop13 = tf.nn.dropout(fire13_fuse, self.keep_prob, name='drop13')
        conv14 = sul.conv_layer('conv14_prob', drop13, filters=channel_size, size=3, stride=1,
                                padding='SAME', relu=False, stddev=0.1)
    
        bilateral_filter_weights = sul.bilateral_filter_layer(
            'bilateral_filter', lidar_input[:, :, :, :3],  # x, y, z
            thetas=[mc.BILATERAL_THETA_A, mc.BILATERAL_THETA_R],
            sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], stride=1)
    
        output_prob = sul.recurrent_crf_layer(
            'recurrent_crf', conv14, bilateral_filter_weights,
            sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], num_iterations=mc.RCRF_ITER,
            padding='SAME'
        )
    
        return output_prob
    
    
    def _voxelnet_feature_module(self):
        
        return None
    
    
