 #include "segment_process.h"
#include "file_trs.h"

float voxel_resolution = 0.75f;//0.75
float seed_resolution = 2.0f;//2.5
float color_importance = 0.0f;
float spatial_importance = 1.0f;//1.0
float normal_importance = 4.0f;//4.0

void step1(string input, pcl::PointCloud<pcl::PointXYZ>::Ptr road, pcl::PointCloud<pcl::PointXYZ>::Ptr remain)
{
	//pcl::console::TicToc tt;


	pcl::PointXYZ point;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile(input, *cloud);

	cerr << "Point cloud data: " << cloud->points.size() << " points" << endl;
	/*
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::io::loadPCDFile(input, *cloud_tmp);
	pcl::copyPointCloud(*cloud_tmp, *cloud);
	*/

	std::cerr << "\nstep1 start..." << std::endl;
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.02); //0.2
	//seg.setMaxIterations(50);

	seg.setInputCloud(cloud);
	seg.setIndices(inliers); // important

	cerr << "before segment" << endl;
	cerr << "check cloud: " << cloud->points.size() << endl;
	seg.segment(*inliers, *coefficients);
	std::cerr << "--------- -----------" << endl;

	if(inliers->indices.size() == 0)
	{
		PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
		return;
	}

	//???????????
	std::cerr << "Model coefficients: " << coefficients->values[0] << " "
		<< coefficients->values[1] << " "
		<< coefficients->values[2] << " "
		<< coefficients->values[3] << std::endl;
	std::cerr << "Model inliers: " << inliers->indices.size() << std::endl;



	std::vector<int> indice_index;
	for (size_t i = 0; i < inliers->indices.size(); ++i)
	{
		road->points.push_back(cloud->points[inliers->indices[i]]);
	}
	for (int i = 0; i < cloud->points.size(); i++)
	{
		indice_index.push_back(i);
	}
	std::vector<int> a;
	for (int m = 0; m < inliers->indices.size(); m++)
	{
		a.push_back(inliers->indices[m]);
	}

	for (int n = 0; n < a.size(); n++)
	{
		std::vector<int>::iterator it = indice_index.begin();//error
		it = it + a[n] - n;
		indice_index.erase(it);
	}

	std::cerr << "step1 >> Done! " << std::endl;
	for (size_t i = 0; i < indice_index.size(); ++i)
	{
		remain->points.push_back(cloud->points[indice_index[i]]);
	}
}


void step2(pcl::PointCloud<pcl::PointXYZ> cloud_in, pcl::PointCloud<pcl::PointXYZ> road, string path, string file)
{
	typedef pcl::PointXYZRGB PointT;
	typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList;
	//???????
	//pcl::console::TicToc tt;
	pcl::PointCloud<PointT>::Ptr input_cloud_ptr(new pcl::PointCloud<PointT>);
	pcl::PCLPointCloud2 input_pointcloud2;
	pcl::copyPointCloud(cloud_in, *input_cloud_ptr);
	PCL_INFO("input cloud size is  : %d", input_cloud_ptr->points.size());

	//PCL_INFO("Done making cloud\n");
	//std::cerr << "Start...\n", tt.tic();
	//??????? ??????????????????????????????  
	bool use_single_cam_transform = false;
	bool use_supervoxel_refinement = false;

	unsigned int k_factor = 0;

	//voxel_resolution is the resolution (in meters) of voxels used??
	//seed_resolution is the average size (in meters) of resulting supervoxels    
	pcl::SupervoxelClustering<PointT> super(voxel_resolution, seed_resolution);
	super.setUseSingleCameraTransform(use_single_cam_transform);
	super.setInputCloud(input_cloud_ptr);
	//Set the importance of color for supervoxels.   
	super.setColorImportance(color_importance);
	//Set the importance of spatial distance for supervoxels.  
	super.setSpatialImportance(spatial_importance);
	//Set the importance of scalar normal product for supervoxels.   
	super.setNormalImportance(normal_importance);
	std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

	PCL_INFO("Extracting supervoxels\n");
	super.extract(supervoxel_clusters);

	PCL_INFO("Getting supervoxel adjacency\n");
	std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
	super.getSupervoxelAdjacency(supervoxel_adjacency);
	//pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud(supervoxel_clusters);

	//LCCP???  
	float concavity_tolerance_threshold = 10.0;// 8 
	float smoothness_threshold = 0.1;//1.0 ????????
	uint32_t min_segment_size = 20;
	uint32_t noise_size = 20;
	bool use_extended_convexity = false;
	bool use_sanity_criterion = false;
	PCL_INFO("Starting Segmentation\n");
	pcl::LCCPSegmentation<PointT> lccp;
	lccp.setConcavityToleranceThreshold(concavity_tolerance_threshold);
	lccp.setSmoothnessCheck(true, voxel_resolution, seed_resolution, smoothness_threshold);
	lccp.setKFactor(k_factor);
	lccp.setInputSupervoxels(supervoxel_clusters, supervoxel_adjacency);
	lccp.setMinSegmentSize(min_segment_size);
	lccp.segment();
	PCL_INFO("Interpolation voxel cloud -> input cloud and relabeling\n");
	pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = super.getLabeledCloud();

	//pcl::PointCloud<pcl::PointXYZL>::Ptr lccp_labeled_cloud = sv_labeled_cloud->makeShared();
	lccp.relabelCloud(*sv_labeled_cloud);
	//SuperVoxelAdjacencyList sv_adjacency_list;
	//lccp.getSVAdjacencyList(sv_adjacency_list);
	//std::map<uint32_t, std::set<uint32_t> > segment_supervoxel_map;
	//lccp.getSegmentToSupervoxelMap(segment_supervoxel_map);

	pcl::PCDWriter writer;
	std::vector<int> labels;
	std::vector<int>::iterator end_unique;
	//std::vector<pcl::PointXYZL>::iterator it;
	pcl::PointCloud<pcl::PointXYZL>::iterator it;
	std::vector<int>::iterator pit;
	int point_num = sv_labeled_cloud->points.size();
	int k = 0;
	for (it = sv_labeled_cloud->begin(); it != sv_labeled_cloud->end(); ++it)
	{
		labels.push_back(it->label);
	}
	std::sort(labels.begin(), labels.end());
	end_unique = std::unique(labels.begin(), labels.end());
	labels.erase(end_unique, labels.end());
	std::stringstream filename;
	char a[100];
	strcpy(a, file.c_str());
	filename << path << "/" << getName(a);
	CreateDir(filename.str());

	std::string ss1;
	ss1 = filename.str() + "/cloud_cluster_0.xyz";
	//cout<<"ss1="<<ss1<<endl;
	road.width = road.points.size();
	road.height = 1;
	road.is_dense = true;
	writer.write<pcl::PointXYZ>(ss1, road, false);

	srand((int)time(NULL));
	int temp = 0;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(road, *cloud_cluster);
	std::vector<int> rgb1;
	for (int i = 0; i<3; i++)
	{
		temp = rand() % 256;
		rgb1.push_back(temp);
	}
	for (int m = 0; m < cloud_cluster->points.size(); m++)
	{
		cloud_cluster->points[m].r = rgb1[0];
		cloud_cluster->points[m].g = rgb1[1];
		cloud_cluster->points[m].b = rgb1[2];
	}
	for (pit = labels.begin(); pit != labels.end(); ++pit)//???????��?label
	{

		std::vector<int> rgb;
		pcl::PCDWriter writer;
		for (int i = 0; i<3; i++)
		{
			temp = rand() % 256;
			rgb.push_back(temp);
		}
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster1(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_xyz(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointXYZRGB single_cloud;
		for (int j = 0; j < point_num; j++)//???????��?point
		{
			if (sv_labeled_cloud->points[j].label == *pit)
			{

				single_cloud.x = sv_labeled_cloud->points[j].x;
				single_cloud.y = sv_labeled_cloud->points[j].y;
				single_cloud.z = sv_labeled_cloud->points[j].z;
				single_cloud.r = rgb[0];
				single_cloud.g = rgb[1];
				single_cloud.b = rgb[2];
				cloud_cluster1->points.push_back(single_cloud);
			}
		}
		if (cloud_cluster1->points.size() < noise_size)
			continue;
		k++;
		std::stringstream ss;
		ss << filename.str() << "/cloud_cluster_" << k << ".xyz";
		cloud_cluster1->points.push_back(single_cloud);
		pcl::copyPointCloud(*cloud_cluster1, *cloud_cluster_xyz);
		cloud_cluster_xyz->width = cloud_cluster_xyz->points.size();
		cloud_cluster_xyz->height = 1;
		cloud_cluster_xyz->is_dense = true;
		writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster_xyz, false);
		*cloud_cluster = *cloud_cluster + *cloud_cluster1;
	}
	cloud_cluster->width = cloud_cluster->points.size();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;
	cout<<"PCDfile:"<<filename.str()<<"/"<<getName(a)<<".pcd"<<" is obtained"<<endl;

	writer.write<pcl::PointXYZRGB>(filename.str() + "/" + getName(a) + ".pcd", *cloud_cluster, false);
	//std::cerr << ">> Done: " << tt.toc() << " ms\n";
}
