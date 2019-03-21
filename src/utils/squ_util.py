#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:19-3-20 下午7:48
# software:PyCharm

import numpy as np
import tensorflow as tf

from util import *

def pretrained_model(mc):
    
    pass

def variable_on_device(name, shape, initializer, trainable=True):
    '''
    :param name:
    :param shape: list of ints
    :param initializer:  initilizer for variable
    :param trainable:
    :return:
    '''
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        
    return var

def variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    '''
    :param name:
    :param shape:
    :param wd: add L2Loss weight decay multiplied by this float.
    if none, weight decay is not added for this variable
    
    :param initializer:
    :param trainable:
    :return:
    '''
    var = variable_on_device(name, shape, initializer, trainable)
    
    if wd is not None and trainable:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    
    return var

def conv_layer(layer_name, inputs, filters, size, stride, padding='SAME',
               use_pretrain=False, freeze=False, xavier=False, relu=True,
               stddev=0.001, bias_init_val=0.0, weight_decay=0.0001):
    '''
    
    :param layer_name:
    :param inputs: inputs tensor
    :param filters: number of output filters.
    :param size: kernel size
    :param stride:
    :param padding: 'SAME' or 'VALID'
    :param use_pretrain:
    :param freeze: if true, then do not train the parameters in this layer.
    :param xavier: whether to use xavier weight initializer or not.
    :param relu: whether to use relu or not.
    :param stddev: standard deviation used for random weight initializer.
    :return:
    '''
    with tf.variable_scope(layer_name) as scope:
        
        # shape [h, w, in, out]
        channels = inputs.get_shape()[3]
        
        if xavier:
            kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
            bias_init = tf.constant_initializer(bias_init_val)
        else:
            kernel_init = tf.truncated_normal_initializer()
            bias_init = tf.constant_initializer(bias_init_val)
        
        # conv
        kernel = variable_with_weight_decay('kernals', shape=[size, size, int(channels), filters],
                                            wd=weight_decay, initializer=kernel_init, trainable=(not freeze))
        bias = variable_on_device('biases', [filters], initializer=bias_init, trainable=(not freeze))
        
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, stride, 1], padding=padding, name= 'convolution')
        conv_bias = tf.nn.bias_add(conv, bias, name='bias_add')
        
        
        # relu
        if relu: out = tf.nn.relu(conv_bias, 'relu')
        else: out = conv_bias
        
        return out
        
def deconv_layer(layer_name, inputs, filters, size, stride, padding='SAME',
                 freeze=False, relu=True, init='trunc_norm',
                 stddev=0.001, bias_init_val=0.0, weight_decay=0.0001):
    
    assert len(size) == 1 or len(size) == 2, \
        'size should be a scalar or an array of size 2.'
    assert len(stride) == 1 or len(stride) == 2, \
        'stride should be a scalar or an array of size 2.'
    assert init == 'xavier' or init == 'bilinear' or init == 'trunc_norm', \
        'init mode not supported {}'.format(init)
    
    # h and w for size and stride
    if len(size) == 1:      size_h, size_w = size[0], size[0]
    else:                   size_h, size_w = size[0], size[1]
    
    if len(stride) == 1:    stride_h, stride_w = stride[0], stride[0]
    else:                   stride_h, stride_w = stride[0], stride[0]
    
    with tf.variable_scope(layer_name) as scope:
        
        batch_size = int(inputs.get_shape()[0])
        height = int(inputs.get_shape()[1])
        width = int(inputs.get_shape()[2])
        channels = int(inputs.get_shape()[3])
        
        if init == 'xavier':
            kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
            bias_init = tf.constant_initializer(bias_init_val)
            
        elif init == 'bilinear':
            kernel_init = np.zeros((size_h, size_w, channels, channels), dtype=np.float32)
            
            # ?
            factor_w = (size_w + 1) // 2
            center_w = (factor_w - 1) if (size_w % 2 == 1) else (factor_w - 0.5)
            og_w = np.reshape(np.arange(size_w), (size_h, -1))
            up_kernel = (1 - np.abs(og_w - center_w) / factor_w)
            
            for c in xrange(channels):
                kernel_init[:, :, c, c] = up_kernel
            
            bias_init = tf.constant_initializer(bias_init_val)
            
        else:
            kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
            bias_init = tf.constant_initializer(bias_init_val)
            
        # w, b
        kernel = variable_with_weight_decay('kernels', shape=[size_h, size_w, filters, channels],
                                            wd=weight_decay, initializer=kernel_init, trainable=(not freeze))
        bias = variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
        
        
        # deconv
        deconv = tf.nn.conv2d_transpose(inputs, kernel,
                                      [batch_size, stride_h*height, stride_w*width, filters],
                                        [1, stride_h, stride_w, 1], padding=padding, name='deconv')
        deconv_bias = tf.nn.bias_add(deconv, bias, name='bias_add')
        
        # relu
        if relu: out = tf.nn.relu(deconv_bias, 'relu')
        else: out = deconv_bias
        
        return out


def conv_bn_layer(inputs, conv_param_name, bn_param_name, scale_param_name,
                   filters, size, stride, padding='SAME', freeze=False, relu=True,
                   conv_with_bias=False, stddev=0.001, bias_init_val=0.0,
                  batch_norm_epsilon=1e-5):
    
    """ Convolution + BatchNorm + [relu] layer. Batch mean and var are treated
    as constant. Weights have to be initialized from a pre-trained model or
    restored from a checkpoint.
    """
    
    with tf.variable_scope(conv_param_name) as scope:
        channels = inputs.get_shape()[3]

        kernel_val = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
        
        if conv_with_bias:
            bias_val = tf.constant_initializer(bias_init_val)
        mean_val = tf.constant_initializer(0.0)
        var_val = tf.constant_initializer(1.0)
        gamma_val = tf.constant_initializer(1.0)
        beta_val = tf.constant_initializer(0.0)
        
        
        # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
        # shape [h, w, in, out]
        kernel =variable_with_weight_decay('kernels', shape=[size, size, int(channels), filters],
                                            wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
        
        if conv_with_bias:
            biases = variable_on_device('biases', [filters], bias_val, trainable=(not freeze))
            
        gamma = variable_on_device('gamma', [filters], gamma_val, trainable=(not freeze))
        beta = variable_on_device('beta', [filters], beta_val, trainable=(not freeze))
        mean = variable_on_device('mean', [filters], mean_val, trainable=False)
        var = variable_on_device('var', [filters], var_val, trainable=False)

        # conv
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, stride, 1], padding=padding, name='convolution')
        
        if conv_with_bias:
            conv = tf.nn.bias_add(conv, biases, name='bias_add')
        
        # batch norm
        conv = tf.nn.batch_normalization(
            conv, mean=mean, variance=var, offset=beta, scale=gamma,
            variance_epsilon=batch_norm_epsilon, name='batch_norm')
        
        # relu
        if relu:
            return tf.nn.relu(conv)
        else:
            return conv
        
def fire_layer(layer_name, inputs, s1x1, e1x1, e3x3,
               stddev=0.001, freeze=False):
    
    '''
    
    :param layer_name:
    :param inputs:
    :param s1x1: number of 1x1 filters in squeeze layer.
    :param e1x1: number of 1x1 filters in expand layer.
    :param e3x3: number of 3x3 filters in expand layer.
    :param stddev:
    :param freeze:
    :return:
    '''
    sq1x1 = conv_layer(layer_name+'squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
                       padding='SAME', freeze=freeze, stddev=stddev)
    ex1x1 = conv_layer(layer_name+'expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
                       padding='SAME', freeze=freeze, stddev=stddev)
    ex3x3 = conv_layer(layer_name+'expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
                       padding='SAME', freeze=freeze, stddev=stddev)
    
    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')


def fire_deconv(layer_name, inputs, s1x1, e1x1, e3x3,
                factors=[1, 2], stddev=0.001, freeze=False):
    
    assert len(factors) == 2, 'factors should be an array of size 2'
    
    ksize_h = factors[0]*2 - factors[0]%2 # 1 -> 1 , 3 -> 5
    ksize_w = factors[1]*2 - factors[1]%2
    
    sq1x1 = conv_layer(layer_name+'squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
                       padding='SAME', freeze=freeze, stddev=stddev)
    deconv = deconv_layer(layer_name+'/deconv', sq1x1, filters=s1x1, size=[ksize_h, ksize_w],
                          stride=factors, padding='SAME', init='bilinear')
    ex1x1 = conv_layer(layer_name + 'expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
                       padding='SAME', freeze=freeze, stddev=stddev)
    ex3x3 = conv_layer(layer_name + 'expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
                       padding='SAME', freeze=freeze, stddev=stddev)
    
    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
    
    
def pooling_layer(layer_name, inputs, size, stride, padding='SAME'):
    
    with tf.variable_scope(layer_name) as scope:
        
        out = tf.nn.max_pool(inputs,
                             ksize=[1, size, size, 1],
                             strides=[1, 1, stride, 1],
                             padding=padding)
        
        return out


def fc_layer(layer_name, inputs, hiddens, flatten=False,
             relu=True, xavier=False, stddev=0.001, bias_init_val=0.0,
             weight_decay=0.0001):
    '''
    :param hiddens: number of (hidden) neurons in this layer
    :param flatten: if true, reshape the input 4d tensor of shape
    (batch, height, weight, channel) into a 2d tensor with shape
    (batch, -1).
    this is used when the input to the fully connected layer is output of a convolutional layer.

    :return:
    '''
    with tf.variable_scope(layer_name) as scope:
        input_shape = inputs.get_shape().as_list()
        
        if flatten:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs = tf.reshape(inputs, [-1, dim])
            
        if xavier:
            kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
            bias_init = tf.constant_initializer(bias_init_val)
        else:
            kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
            bias_init = tf.constant_initializer(bias_init_val)
            
        
        weight = variable_with_weight_decay('weights', shape=[dim, hiddens],
                                            wd=weight_decay, initializer=kernel_init)
        bias = variable_on_device('biases', [hiddens], initializer=bias_init)
    
        outputs = tf.nn.bias_add(tf.matmul(inputs, weight), bias)
        
        if relu: outputs = tf.nn.relu(outputs, 'relu')
        
        num_flops = 2*dim*hiddens + hiddens
        
        return outputs
   
    
def recurrent_crf_layer(layer_name, inputs, bilateral_filters, size=[3, 5],
                        num_iterations=1, padding='SAME',
                        num_class=5, bi_filter_coef=0.1, ang_filter_coef=0.02,
                        ang_theta_a=np.array([.9, .9, .6, .6]),
                        bilateral_theta_a=np.array([.9, .9, .6, .6]),
                        bilateral_theta_r=np.array([.015, .015, .01, .01])):
    
    assert (num_iterations >= 1, 'number of iterations should >= 1')
    
    with tf.variable_scope(layer_name) as scope:
        
        # initialize compatibilty matrices
        compat_kernel_init = tf.constant(np.reshape(
            np.ones((num_class, num_class)) - np.identity(num_class),
            newshape=[1, 1, num_class, num_class]
        ), dtype=tf.float32)
        
        bi_compat_kernel = variable_on_device(
            name='bilateral_compatibility_matrix',
            shape=[1, 1, num_class, num_class],
            initializer=compat_kernel_init*bi_filter_coef,
            trainable=True
        )
        
        angular_compat_kernel = variable_on_device(
            name='angular_compatibility_matrix',
            shape=[1, 1, num_class, num_class],
            initializer=compat_kernel_init*ang_filter_coef,
            trainable=True
        )
        
        #
        condensing_kernel = tf.constant(
            condensing_matrix(size[0], size[1], num_class),
            dtype=tf.float32,
            name='condensing_kernel'
        )
        
        angular_filters = tf.constant(
            angular_filter_kernel(size[0], size[1], num_class,
                                  ang_theta_a ** 2),
            dtype=tf.float32,
            name='angular_kernel'
        )
        
        bi_angular_filters = tf.constant(
            angular_filter_kernel(size[0], size[1], num_class,
                                  bilateral_theta_a ** 2),
            dtype=tf.float32,
            name='bi_angular_kernel'
        )
        
        for iter in range(num_iterations):
            
            unary = tf.nn.softmax(inputs, dim=-1, name='unary_term_at_iter_{}'.format(iter))
            
            ang_output, bi_output = locally_connected_layer(
                'message_passing_iter_{}'.format(iter), unary,
                bilateral_filters, angular_filters, bi_angular_filters,
                condensing_kernel, size=size, padding=padding
            )
            
            ang_output = tf.nn.conv2d(ang_output, angular_compat_kernel, strides=[1, 1, 1, 1],
                                      padding='SAME', name='angular_compatibility_transformation')
            bi_output = tf.nn.conv2d(bi_output, bi_compat_kernel, strides=[1, 1, 1, 1],
                                     padding='SAME', name='bilateral_comptibility_transformation')
    
            pairwise = tf.add(ang_output, bi_output,
                              name='pairwise_term_at_iter_{}'.format(iter))
            outputs = tf.add(unary, pairwise,
                             name='eneray_at_iter_{}'.format(iter))
            
            inputs = outputs
        
        return outputs


def locally_connected_layer(
        layer_name, inputs, bilateral_filters,
        angular_filters, bi_angular_filters, condensing_kernel,
        sizes=[3, 5], padding='SAME', mask=[]):
    
    """
    Locally connected layer with non-trainable filter parameters)
    
    Returns:
          ang_output: output tensor filtered by anguler filter with shape
              [batch_size, zenith, azimuth, num_class].
          bi_output: output tensor filtered by bilateral filter with shape
              [batch_size, zenith, azimuth, num_class].
    """
    assert padding == 'SAME', 'only support SAME padding strategy'
    assert sizes[0] % 2 == 1 and sizes[1] % 2 == 1, \
        'Currently only support odd filter size.'
    
    size_z, size_a = sizes
    pad_z, pad_a = size_z // 2, size_a // 2
    
    half_filter_dim = (size_z * size_a) // 2
    batch, zenith, azimuth, in_channel = inputs.shape.as_list()
    
    with tf.variable_scope(layer_name) as scope:
        
        # message passing
        ang_output = tf.nn.conv2d(
            inputs, angular_filters, [1, 1, 1, 1], padding=padding,
            name='angular_filtered_term'
        )
        
        bi_ang_output = tf.nn.conv2d(
            inputs, bi_angular_filters, [1, 1, 1, 1], padding=padding,
            name='bi_angular_filtered_term'
        )
        
        condensed_input = tf.reshape(
            tf.nn.conv2d(
                inputs * mask, condensing_kernel, [1, 1, 1, 1], padding=padding,
                name='condensed_prob_map'
            ),
            [batch, zenith, azimuth, size_z * size_a - 1, in_channel]
        )
        
        bi_output = tf.multiply(
            tf.reduce_sum(condensed_input * bilateral_filters, axis=3),
            mask, name='bilateral_filtered_term'
        )
        bi_output *= bi_ang_output
    
    return ang_output, bi_output



def bilateral_filter_layer(
        layer_name, inputs,
        thetas=[0.9, 0.01], sizes=[3, 5], stride=1,
        padding='SAME', num_class=5,
        bilateral_theta_a=np.array([.9, .9, .6, .6]),
        bilateral_theta_r=np.array([.015, .015, .01, .01])):
    
    """
    Computing pairwise energy with a bilateral filter for CRF.

    Returns:
      out: bilateral filter weight output with size
          [batch_size, zenith, azimuth, sizes[0]*sizes[1]-1, num_class]. Each
          [b, z, a, :, cls] represents filter weights around the center position
          for each class.
    """
    
    assert padding == 'SAME', 'currently only supports "SAME" padding stategy'
    assert stride == 1, 'currently only supports striding of 1'
    assert sizes[0] % 2 == 1 and sizes[1] % 2 == 1, \
        'Currently only support odd filter size.'
    
    theta_a, theta_r = thetas
    size_z, size_a = sizes
    pad_z, pad_a = size_z // 2, size_a // 2
    half_filter_dim = (size_z * size_a) // 2
    batch, zenith, azimuth, in_channel = inputs.shape.as_list()
    
    # assert in_channel == 1, 'Only support input channel == 1'
    
    with tf.variable_scope(layer_name) as scope:
        
        condensing_kernel = tf.constant(
            condensing_matrix(size_z, size_a, in_channel),
            dtype=tf.float32,
            name='condensing_kernel'
        )
        
        condensed_input = tf.nn.conv2d(
            inputs, condensing_kernel, [1, 1, stride, 1], padding=padding,
            name='condensed_input'
        )
        

        diff_x = tf.reshape(inputs[:, :, :, 0],
                            [batch, zenith, azimuth, 1]) - condensed_input[:, :, :, 0::in_channel]
        diff_y = tf.reshape(inputs[:, :, :, 1],
                            [batch, zenith, azimuth, 1]) - condensed_input[:, :, :, 1::in_channel]
        diff_z = tf.reshape(inputs[:, :, :, 2],
                            [batch, zenith, azimuth, 1]) - condensed_input[:, :, :, 2::in_channel]
        
        bi_filters = []
        
        for cls in range(num_class):
            theta_a = mc.BILATERAL_THETA_A[cls]
            theta_r = mc.BILATERAL_THETA_R[cls]
            bi_filter = tf.exp(-(diff_x ** 2 + diff_y ** 2 + diff_z ** 2) / 2 / theta_r ** 2)
            bi_filters.append(bi_filter)
            
        out = tf.transpose(
            tf.stack(bi_filters),
            [1, 2, 3, 4, 0],
            name='bilateral_filter_weights'
        )
    
    return out





















