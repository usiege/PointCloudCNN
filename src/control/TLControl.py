#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:2019/5/15 下午8:50
# software:PyCharm

import tensorflow as tf
import tensorlayer as tl

def train():
    sess = tf.InteractiveSession()
    
    # 准备数据
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 784))
    
    # 定义 placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
    
    # 定义模型
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=800,
                                   act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=800,
                                   act=tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   name='output_layer')
    # 定义损失函数和衡量指标
    # tl.cost.cross_entropy 在内部使用 tf.nn.sparse_softmax_cross_entropy_with_logits() 实现 softmax
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_, name='cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    
    # 定义 optimizer
    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
    
    # 初始化 session 中的所有参数
    # tl.layers.initialize_global_variables(sess)
    init = tf.global_variables_initializer()
    # init = tf.initialize_all_variables()
    sess.run(init)
    
    print("print")
    # 列出模型信息
    network.print_params()
    network.print_layers()
    
    print("train")
    # 训练模型
    tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
                 acc=acc, batch_size=500, n_epoch=500, print_freq=5,
                 X_val=X_val, y_val=y_val, eval_train=False)
    
    # 评估模型
    tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)
    
    # 把模型保存成 .npz 文件
    tl.files.save_npz(network.all_params, name='model.npz')
    sess.close()
    print("end!")
    
if __name__ == "__main__":
    train()