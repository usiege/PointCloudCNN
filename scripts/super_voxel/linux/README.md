sence_segmentation_rgb.cpp中需要设置的路径有三个：

一、input_path：需要分割的.txt文件的上级目录。其中，.txt文件数据为4列，前三列分别为x、y、z轴方向上的坐标值。
二、input_trs_path：用户自定义文件夹路径，该文件夹用来保存转化后的.xyz文件。
三、result_path：用户自定义文件夹路径，该文件夹用来保存分割后结果数据。其中*.xyz文件为单个物体，*.pcd为带假彩色分割后场景。