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
from config.config import *

import time
import tensorflow as tf

FLAGS = arg_config()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch

BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


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
    
        self.mc = None
        self.saver = None

        self.ops = {}
        self.net = PointCloud('')

        # self._net_did_load()

    def load_net(self, mc):
        self.mc = mc
        self._net_did_load(is_training=True, train_enable=True, val_enable=False)
        
    def _net_did_load(self, is_training=True, train_enable=True, val_enable=False, 
        new_train=True):
        
        LOG_DIR = FLAGS.log_dir
        TRAIN_DIR = os.path.join(LOG_DIR, 'train')
        EVAL_DIR = os.path.join(LOG_DIR, 'eval')
        TEST_DIR = os.path.join(LOG_DIR, 'test')
        CHECKPOINT_DIR = os.path.join(LOG_DIR, 'model')
        
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        
        self.model = Model(self.mc, FLAGS)
        self.net.model = self.model
        
        with tf.Graph().as_default() as graph:

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
                self.saver = saver
            
            with tf.device('/gpu:' + str(GPU_INDEX)):
                
                with tf.Session() as sess:
                    # Create a session
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    config.allow_soft_placement = True
                    config.log_device_placement = True
                    sess = tf.Session(config=config)

                    # if old restore model
                    global_step = 0
                    if not new_train:
                        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
                        ckpt_path = ckpt.model_checkpoint_path
                        if ckpt and ckpt_path:
                            saver.restore(sess, ckpt_path)
                            global_step = ckpt_path.split('/')[-1].split('-')[-1]

                    # Add summary writers
                    merged = tf.summary.merge_all()
                    train_writer = tf.summary.FileWriter(TRAIN_DIR, graph)
                    eval_writer = tf.summary.FileWriter(EVAL_DIR, graph)
                    self.ops["merged"] = merged
                    
                    # Init variables
                    init = tf.global_variables_initializer()
                    sess.run(init, {is_training_pl: is_training})
                    
                    for epoch in range(global_step, MAX_EPOCH):
                        self.model.log_string('**** EPOCH %03d ****' % (epoch))
                        sys.stdout.flush()
                        
                        if train_enable:
                            self._train_one_epoch(sess, train_writer)
                        if val_enable:
                            self._eval_one_epoch(sess, eval_writer)
                        
                        # Save the variables to disk.
                        if epoch % 10 == 0:
                            save_path = saver.save(sess, os.path.join(CHECKPOINT_DIR, "model.ckpt"),global_step=epoch)
                            self.model.log_string("Model saved in file: %s" % save_path)
    
    def _train_one_epoch(self, sess, writer):
        
        # GPU_INDEX = GPU_INDEX
        is_training = True
        
        ops = self.ops
        loss_sum = 0
        total_correct = 0
        total_seen = 0

        # load data for network
        self.model.load_imageset('train')
        data_total_size = len(self.model.image_set)
        num_batches = data_total_size // BATCH_SIZE
        
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
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum(pred_val == labels)
                
                total_correct += correct
                total_seen += (BATCH_SIZE*NUM_POINT)
                
                if batch_idx % 100 == 0:
                    loss_log = 'loss when batch_idx : %f(loss)/%d(batch)' % ((loss_val / float(BATCH_SIZE)), batch_idx)
                    self.model.log_string(loss_log)
                if batch_idx % 10 == 0:
                    print('Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
                    print('loss -----> %f' % (loss_val / float(BATCH_SIZE)))

            self.model.log_string('mean loss: %f' % (loss_sum / float(num_batches)))
            self.model.log_string('accuracy ------> %f' % (total_correct / float(total_seen)))

    def _eval_one_epoch(self, sess, writer):
    
        ops = self.ops
        is_training = False
        loss_sum = 0
        
        # load data for network
        self.model.load_imageset(process='eval')
        data_total_size = len(self.model.image_set)
        num_batches = data_total_size // BATCH_SIZE

        total_correct = 0
        total_seen = 0
        num_classes = self.model.mc.NUM_CLASS

        total_seen_class = np.zeros(num_classes)
        total_correct_class = np.zeros(num_classes)
        
        self.model.log_string('----------- evaluation ---------')
        
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
                
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum(pred_val == labels)
                total_correct += correct
                total_seen += BATCH_SIZE*NUM_POINT
                loss_sum += loss_val*BATCH_SIZE
                
                for i in range(labels.shape[0]):
                    for j in range(labels.shape[1]):
                        l = labels[i, j]
                        total_seen_class[l] += 1
                        total_correct_class[l] += (pred_val[i, j] == l)
                        
                if batch_idx % 10 == 0:
                    print('detection for batch %d/%d' % (batch_idx, num_batches))
            

            self.model.log_string('eval mean loss: %f' % (loss_sum / float(total_seen / NUM_POINT)))
            self.model.log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
            self.model.log_string('eval avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / np.array(total_seen_class,
                                                                 dtype=np.float))))
        


def train():
    print('this is training process ...\n')
    

def eval():
    print('this is evaluation process ...\n')
    

def test():
    print('this is testing process ...\n')
    



if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    ct = Controller()
    mc = lidar_2d_config()
    ct.load_net(mc)

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
