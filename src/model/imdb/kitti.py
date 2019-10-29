
class Kitti(object):
    """docstring for Kitti"""
    
    def __init__(self, arg):
        super(Kitti, self).__init__()
        self.arg = arg
        
        self._image_set = arg.image_set
        self._data_root_path = os.path.join(self._data_root_path, '')
        self._imageset_root_path = os.path.join(self._imageset_root_path, 'ImageSets')
        
        self._imageset_idx = self._load_imageset_idx()
        
        self._data_crop_path = os.path.join(self._data_root_path, 'crop')
        self._data_training_path = os.path.join(self._data_root_path, 'training')
        self._data_testing_path = os.path.join(self._data_root_path, 'testing')
        
        self._kitti_sub_dirs = ['calib', 'image_2', 'label_2', 'velodyne']
    
    @property
    def data_crop_path(self):
        return self._data_crop_path
    
    @property
    def data_training_path(self):
        return self._data_training_path
    
    @property
    def data_testing_path(self):
        return self._data_testing_path
    
    
