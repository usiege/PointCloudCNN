import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import sys

sys.path.append(os.path.join(ROOT_DIR, 'imdb'))
sys.path.append(os.path.join(ROOT_DIR, 'config'))
sys.path.append(os.path.join(ROOT_DIR, 'net'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from imdb.model import *
from net.pointcloud import *

from config.lidar_2d_config import *
# from config import config

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--p', default='train', help='for process use ["train","eval","test"]')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096 * 8, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()

import tensorflow as tf

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch

BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

# MAX_NUM_POINT = 4096

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


# def feed_dict(data):
#     return {}

def data_to_feeddata(batch_size, origin_data):
    data = np.reshape(origin_data, (batch_size, origin_data.shape[1] * origin_data.shape[2], origin_data.shape[3]))
    
    inputs, outputs = data[:, 0:NUM_POINT, 0:5], data[:, 0:NUM_POINT, 5]
    return inputs, outputs


class Controller(object):
    """docstring for Controller"""
    
    def __init__(self):
        super(Controller, self).__init__()
        
        mc = lidar_2d_config()  # lidar_2d
        self.model = Model(mc, FLAGS)
        self.net = PointCloud(self.model)
        
        self.ops = {}
        
        self._net_did_load()
    
    def _net_did_load(self):
        self._tf_process()
    
    def _tf_process(self, is_training=True, process='train'):
        
        LOG_DIR = FLAGS.log_dir
        
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(GPU_INDEX)):
                pts_pl, labels_pl = self.net.get_input_tensor()
                is_training_pl = self.net.get_is_training_tensor()
                
                self.ops["pts_pl"] = pts_pl
                self.ops["labels_pl"] = labels_pl
                self.ops["is_training_pl"] = is_training_pl
                
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)
                self.ops["step"] = batch
                
                # get model and loss
                pred = self.net.get_net(pts_pl, is_training_pl)
                loss = self.net.get_loss(pred, labels_pl)
                tf.summary.scalar('loss', loss)
                self.ops["pred"] = pred
                self.ops["loss"] = loss
                
                correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
                tf.summary.scalar('accuracy', accuracy)
                
                # Get training operator
                learning_rate = get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)
                self.ops["train_op"] = train_op
                
                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()
            
            with tf.device('/gpu:' + str(GPU_INDEX)):
                with tf.Session() as sess:
                    # Create a session
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    config.allow_soft_placement = True
                    config.log_device_placement = True
                    sess = tf.Session(config=config)
                    
                    # Add summary writers
                    merged = tf.summary.merge_all()
                    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
                    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
                    self.ops["merged"] = merged
                    
                    # Init variables
                    init = tf.global_variables_initializer()
                    sess.run(init, {is_training_pl: is_training})
                    
                    for epoch in range(MAX_EPOCH):
                        self.model.log_string('**** EPOCH %03d ****' % (epoch))
                        sys.stdout.flush()
                        
                        self._train_one_epoch(sess, train_writer)
                        self._eval_one_epoch(sess, train_writer)
                        
                        # Save the variables to disk.
                        if epoch % 1000 == 0:
                            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                            self.model.log_string("Model saved in file: %s" % save_path)
    
    def _train_one_epoch(self, sess, writer):
        
        # GPU_INDEX = GPU_INDEX
        
        data_total_size = len(self.model.image_set)
        num_batches = data_total_size // BATCH_SIZE
        is_training = True
        
        ops = self.ops
        
        loss_sum = 0
        
        with tf.device("/gpu:" + str(GPU_INDEX)):
            for batch_idx in range(num_batches):
                batch_data = self.model.read_batch()
                pointclouds, labels = data_to_feeddata(BATCH_SIZE, batch_data)
                feed_dict = {ops['pts_pl']: pointclouds,
                             ops['labels_pl']: labels,
                             ops['is_training_pl']: is_training, }
                summary, step, _, loss_val, pred_val = sess.run(
                    [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                    feed_dict=feed_dict)
                
                writer.add_summary(summary, step)
                loss_sum += loss_val
                
                if batch_idx % 100 == 0:
                    loss_log = 'loss when batch_idx : %f(loss)/%d(batch)' % (loss_val / float(BATCH_SIZE)), batch_idx
                    self.model.log_string(loss_log)
                if batch_idx % 10 == 0:
                    print('Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
                    print('loss -----> %f' % (loss_val / float(BATCH_SIZE)))
    
            self.model.log_string('mean loss: %f' % (loss_sum / float(num_batches)))


def _eval_one_epoch(self, sess, writer):
    pass


def feed_ops(self, process='train'):
    return {}


def train(self):
    print('this is training process ...\n')
    self._tf_process(process='train')


def eval(self):
    print('this is evaluation process ...\n')
    self._tf_process(process='val')


def test(self):
    print('this is testing process ...\n')
    self._tf_process(is_training=False, process='test')


# ct = Controller('controller')

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    ct = Controller()
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if False:
        with tf.Graph().as_default():
            with tf.device("/gpu:0"):
                a = tf.placeholder(tf.float32, shape=(32, 4096, 5))
                mc = lidar_2d_config()
                model = Model(mc, FLAGS)
                pc = PointCloud(model)
                net = pc.get_net(a, tf.constant(True))
                
                with tf.Session() as sess:
                    tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
                    sess = tf.Session(config=tfconfig)
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    start = time.time()
                    for i in range(100):
                        print(i)
                        sess.run(net, feed_dict={a: np.random.rand(32, 4096, 5)})
                    print(time.time() - start)
