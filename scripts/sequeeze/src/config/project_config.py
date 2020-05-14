
import tensorflow as tf

DEBUG = True

DATA_WORKING_PATH       = "npy360_full"
TRAINING_TOTAL_COUNT    = 27000
EVALUATION_TOTAL_COUNT  = 2000

TESTING_MODEL_PATH = "../scripts/log/train8/model.ckpt-29000"

MODEL_TRAIN_DIR     = "../scripts/log/train8_360"
MODEL_CHECKPOINT    = MODEL_TRAIN_DIR + "/model.ckpt-26000"

FLAGS = tf.app.flags.FLAGS

# ../scripts/log/train_finetune/model.ckpt-21000
    # ../data/SqueezeSeg/model.ckpt-23000
    
tf.app.flags.DEFINE_string('train_dir', MODEL_TRAIN_DIR,
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('checkpoint_path', MODEL_TRAIN_DIR,
                           """Path to the training checkpoint.""")

tf.app.flags.DEFINE_string('input_path', '../data/test2/npy/*',
                           """Input lidar scan to be detected. Can process glob input such as """
                           """./data/samples/*.npy or single input.""")

tf.app.flags.DEFINE_string( 'checkpoint', MODEL_CHECKPOINT,
                            """Path to the model parameter file.""")


# ../scripts/log/answers/
tf.app.flags.DEFINE_string('out_dir', '../scripts/log/answers_8/',
                           """Directory to dump output.""")

tf.app.flags.DEFINE_string('eval_dir', '../scripts/log/eval_val8',
                           """Directory where to write event logs """)


tf.app.flags.DEFINE_string('pretrained_model_path', '../data/SqueezeNet/squeezenet_v1.1.pkl',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_string('data_path', '../data/', """Root directory of data""")
tf.app.flags.DEFINE_string('dataset', 'KITTI', """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('net', 'squeezeSeg', """Neural net architecture. """)











