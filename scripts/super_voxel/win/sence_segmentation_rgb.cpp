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
#include <direct.h> 
#include <io.h>   
#include <file_trs.h>
#include <segment_process.h>
int
main(int argc, char** argv)
{
	
	vector<string> files;
	vector<string> filename;
	string input_path = "//"D:\\文档\\自动化所\\forsave";";
	string input_trs_path = "D:\\文档\\自动化所\\xyz";
	string result_path = "C:\\Users\\zimu\\Desktop\\show";
	float voxel_resolution = 0.43, seed_resolution = 1.32;
	uint32_t min_segment_size = 1000;
	pcl::PCDWriter writer;
	set_para(voxel_resolution, seed_resolution, min_segment_size);
	/*
	if (file_xyz(input_path, input_trs_path) == -1)
	{
		cerr << "can not open file" << endl;
		return -1;
	}
	*/
	getFiles(input_trs_path, files, filename);
	
	for (vector<string>::iterator it = files.begin(), pit = filename.begin(); it != files.end(), pit != filename.end(); it++,pit++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr road(new pcl::PointCloud<pcl::PointXYZ>), remain(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ> cloud;
		pcl::io::loadPCDFile(*it, *remain);
		step2(*remain, *road, result_path, *pit);
		//step1(*it,road,remain);
		//step2(*remain, *road, result_path, *pit);	
	}
	return (0);
}