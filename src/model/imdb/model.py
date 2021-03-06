import numpy as np
import os


def data_to_feeddata(batch_size, origin_data):
    return None


class Model(object):
    """docstring for Model"""
    
    def __init__(self, arg, FLAGS):
        super(Model, self).__init__()
        # config for model
        self._mc = arg
        
        # if arg == 'lidar_2d':
        #     self._mc = lidar_2d_config()
        # elif arg == 'kitti':
        #     self._mc = kitti_config()
        
        # project flages from outside of script
        self._FLAGS = FLAGS
        # write flags into file
        self._write_parser()
        
        # data root path
        self._data_root_path = arg.DATA_PATH
        # imageset file root path 'root/ImageSets/...''
        self._imageset_root_path = arg.IMAGESET_PATH
        
        # imageset data idxs
        self._imageset_idx = []
        
        # batch reader
        self._permutation_idx = []
        self._cur_idx = 0
        self._batch_size = FLAGS.batch_size
        
        # transfer for original data
        self._transfer = Transfer(arg)
        # statistic tool for data
        self._statistic = Statistic(arg)
        
        
    @property
    def FLAGS(self):
        return self._FLAGS
        
    @property
    def mc(self):
        return self._mc
    
    @property
    def data_root_path(self):
        return self._data_root_path
    
    @property
    def image_set(self):
        return self._imageset_idx


# ############################## model
    
    def load_imageset(self, process='train'):
        self._imageset_idx = self._load_imageset_idx(process=process)
        self._permutation_idx = self._imageset_idx
        self._cur_idx = 0


    def _load_imageset_idx(self, process='train'):

        if process == 'train':
            image_set_file = os.path.join(self._imageset_root_path, 'train.txt')
        elif process == 'eval':
            image_set_file = os.path.join(self._imageset_root_path, 'val.txt')
        elif process == 'test':
            image_set_file = os.path.join(self._imageset_root_path, 'test.txt')
        else:
            image_set_file = os.path.join(self._imageset_root_path, 'all.txt')

        assert os.path.exists(image_set_file), \
            'File does not exist: {}'.format(image_set_file)
    
        with open(image_set_file) as f:
            image_idx = [x.strip() for x in f.readlines()]
    
        return image_idx
    
    # # None
    # def load_collection(self, process='all'):
    #     return []


    def _shuffle_image_idx(self):
        self._permutation_idx = [self._imageset_idx[i] \
                          for i in np.random.permutation(np.arange(len(self._imageset_idx)))]
        self._cur_idx = 0
    
    def _lidar_2d_path_at(self, idx):
        lidar_2d_path = os.path.join(self._data_root_path, idx + '.npy')
        assert os.path.exists(lidar_2d_path), \
            'File does not exist: {}'.format(lidar_2d_path)
        
        return lidar_2d_path
    
    
    def read_data(self, path):
        record = np.load(path).astype(np.float32, copy=False)
        return record
    
    
    def read_batch(self, process='train', shuffle=True):
        
        image_idx = self._imageset_idx
        
        if shuffle:
            if self._cur_idx + self._batch_size >= len(image_idx):
                self._shuffle_image_idx()
            batch_idx = self._permutation_idx[self._cur_idx: self._cur_idx + self._batch_size]
            self._cur_idx += self._batch_size
        else:
            if self._cur_idx + self._batch_size >= len(image_idx):
                batch_idx = image_idx[self._cur_idx:] + \
                            image_idx[:self._cur_idx + self._batch_size - len(image_idx)]
                self._cur_idx += self._batch_size - len(self._image_idx)
            else:
                batch_idx = image_idx[self._cur_idx: self._cur_idx + self._batch_size]
                self._cur_idx += self._batch_size
        
        batch_data = []
        
        for idx in batch_idx:
            # load record
            path = self._lidar_2d_path_at(idx)
            record = self.read_data(path)
            batch_data.append(record)
        
        return np.array(batch_data)
    
    
    def read_all_data(self, shuffle=False):
        image_idx = self._imageset_idx
        
        if shuffle:
            self._shuffle_image_idx()
        
        data = []
        for idx in image_idx:
            path = self._lidar_2d_path_at(idx)
            record = self.read_data(path)
            data.append(record)
    
        return np.array(data)
    
    def _write_parser(self):
        FLAGS = self.FLAGS
        log_dir = FLAGS.log_dir
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        log_txt_path = os.path.join(log_dir, 'log.txt')
        
        with open(log_txt_path, 'w') as f:
            f.write(str(FLAGS) + '\n')
            f.flush()
            
            # self._log_file_out = f
            self._log_txt_path = log_txt_path
    
    def log_string(self, out_str):
        with open(self._log_txt_path, 'a+') as f:
            f.write(out_str + '\n')
            f.flush()
            print('log out:' + out_str + '\n')


# load data
class Transfer(object):
    """docstring for Loder"""
    
    def __init__(self, arg):
        super(Transfer, self).__init__()
        self.arg = arg


# for statistic
class Statistic(object):
    """docstring for Statistic"""
    
    def __init__(self, arg):
        super(Statistic, self).__init__()
        self.arg = arg
        
    
    def statistic_data(self, data):
        print (np.shape(data))
        
        


def main():
    print('this is a test script!')


if __name__ == '__main__':
    main()
