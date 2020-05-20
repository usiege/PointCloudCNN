#include <stdlib.h>  
#include <cmath>  
#include <limits.h>  
#include <boost/format.hpp>  
#include <pcl/console/parse.h>  
#include <pcl/visualization/pcl_visualizer.h>  
#include <pcl/visualization/point_cloud_color_handlers.h>  
#include <pcl/filters/passthrough.h>  
#include <pcl/segmentation/supervoxel_clustering.h>  
#include <pcl/segmentation/lccp_segmentation.h>  
#include <algorithm>
#include <iostream>
#include <vector>
#include <pcl/console/time.h>
#include <conio.h>
#include <ctime>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/console/time.h>
#include <conio.h>
#include <ctime>
#include <string>
#include <io.h>
using namespace std;
void set_para(float voxel_resolution, float seed_resolution, uint32_t min_segment_size);
/*ransac算法提取平面*
cloud_in:输入点云
road:提取出的路面点云
remain:剩余点云
*/
void step1(string input, pcl::PointCloud<pcl::PointXYZ>::Ptr road, pcl::PointCloud<pcl::PointXYZ>::Ptr remain);
/*跨凸边增长算法提取汽车等物体
cloud_in:输入点云
road:提取出的路面点云
path:结果保存路径
file:结果保存的文件夹名字
*/
void step2(pcl::PointCloud<pcl::PointXYZ> cloud_in, pcl::PointCloud<pcl::PointXYZ> road, string path, string file);


