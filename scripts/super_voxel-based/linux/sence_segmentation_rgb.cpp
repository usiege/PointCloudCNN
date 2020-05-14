#include <stdlib.h>  
#include <cmath>  
#include <limits.h>  
#include <boost/format.hpp>  
#include <pcl/console/parse.h>  
// #include <pcl/visualization/pcl_visualizer.h>  
// #include <pcl/visualization/point_cloud_color_handlers.h>  
#include <pcl/filters/passthrough.h>  
#include <pcl/segmentation/supervoxel_clustering.h>  
#include <pcl/segmentation/lccp_segmentation.h>  
#include <algorithm>
#include <iostream>
#include <vector>
#include <pcl/console/time.h>
//#include <conio.h>
#include <ctime>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/console/time.h>
#include <ctime>
#include <string>
//#include <direct.h> 
#include <sys/io.h>   
#include "file_trs.h"
#include "segment_process.h"

int main(int argc, char** argv)
{
	
	vector<string> files;
	vector<string> filename;


	//original
	string input_path = "/media/charles/exFAT/working/PointCloud/scripts/super_voxel-based/data/data_txt";
	//des
	string input_trs_path = "/media/charles/exFAT/working/PointCloud/scripts/super_voxel-based/data/data_xyz";
	string result_path = "/media/charles/exFAT/working/PointCloud/scripts/super_voxel-based/data/result";

	float voxel_resolution = 0.75f;//0.75
	float seed_resolution = 2.0f;//2.5
	float spatial_importance = 1.0f;//1.0
	float normal_importance = 4.0f;//4.n

	cout << "segmentation test" << endl;

	pcl::PCDWriter writer;

	//*
	if (file_xyz(input_path, input_trs_path) == -1)
	{

		cerr << "can not open file" << input_path<<" and "<<input_trs_path<< endl;
		return -1;
	}
	//*/

	
	getFiles(input_trs_path, files, filename);
	cout << "input string path update done!" << endl;

	int i = 0;
	for (vector<string>::iterator it = files.begin(), pit = filename.begin(); 
	it != files.end(), pit != filename.end(); it++,pit++)
	{
		cout << "files: " << *it << endl;
		cout << "filename: " << *pit << endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr road(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr remain(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ> cloud;

		pcl::io::loadPCDFile(*it, *remain);
		//step1(*it,road,remain);
		//cout << "step1 done!" << endl;

		step2(*remain, *road, result_path, *pit);
		cout << "step2 done!" << endl;
		i++;
		if(i == 1) break;
	}
	
	cout << "END!" << endl;
	return (0);
}
