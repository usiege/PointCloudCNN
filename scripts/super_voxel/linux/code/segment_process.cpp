#include <segment_process.h>
#include <file_trs.h> 

float voxel_resolution=0.75, seed_resolution=2.0, spatial_importance=1.0, normal_importance=5.0, color_importance = 0.0;
uint32_t min_segment_size=100;
void set_para(float voxel_resolution_, float seed_resolution_,  uint32_t min_segment_size_)
{
	voxel_resolution = voxel_resolution_;//0.75
	seed_resolution = seed_resolution_;//2.5
	min_segment_size = min_segment_size_;
}
void step1(string input, pcl::PointCloud<pcl::PointXYZ>::Ptr road, pcl::PointCloud<pcl::PointXYZ>::Ptr remain)
{
	//pcl::console::TicToc tt;
	pcl::PointXYZ point;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile(input, *cloud);
	//std::cerr << "start...\n", tt.tic();
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	std::vector<int> indice_index;
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.5);
	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);
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
	//std::cerr << ">> Done: " << tt.toc() << " ms\n";
	for (size_t i = 0; i < indice_index.size(); ++i)
	{
		remain->points.push_back(cloud->points[indice_index[i]]);
	}
}
void step2(pcl::PointCloud<pcl::PointXYZ> cloud_in, pcl::PointCloud<pcl::PointXYZ> road, string path, string file)
{
	typedef pcl::PointXYZRGB PointT;
	typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList;
	//输入点云  
	//pcl::console::TicToc tt;
	pcl::PointCloud<PointT>::Ptr input_cloud_ptr(new pcl::PointCloud<PointT>);
	pcl::PCLPointCloud2 input_pointcloud2;
	pcl::copyPointCloud(cloud_in, *input_cloud_ptr);
	//PCL_INFO("Done making cloud\n");
	//std::cerr << "Start...\n", tt.tic();
	//超体聚类 参数依次是粒子距离、晶核距离、颜色容差、  
	bool use_single_cam_transform = false;
	bool use_supervoxel_refinement = false;

	unsigned int k_factor = 0;

	//voxel_resolution is the resolution (in meters) of voxels used、seed_resolution is the average size (in meters) of resulting supervoxels    
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

	//PCL_INFO("Extracting supervoxels\n");
	super.extract(supervoxel_clusters);

	//PCL_INFO("Getting supervoxel adjacency\n");
	std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
	super.getSupervoxelAdjacency(supervoxel_adjacency);
	//pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud(supervoxel_clusters);

	//LCCP分割  
	float concavity_tolerance_threshold = 8.0;// 8 
	float smoothness_threshold = 0.1;//1.0 无太大影响
	uint32_t noise_size = cloud_in.points.size() / min_segment_size;
	bool use_extended_convexity = false;
	bool use_sanity_criterion = false;
	//PCL_INFO("Starting Segmentation\n");
	pcl::LCCPSegmentation<PointT> lccp;
	lccp.setConcavityToleranceThreshold(concavity_tolerance_threshold);
	lccp.setSmoothnessCheck(true, voxel_resolution, seed_resolution, smoothness_threshold);
	lccp.setKFactor(k_factor);
	lccp.setInputSupervoxels(supervoxel_clusters, supervoxel_adjacency);
	lccp.setMinSegmentSize(noise_size);
	lccp.segment();
	//PCL_INFO("Interpolation voxel cloud -> input cloud and relabeling\n");
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
	pcl::PointCloud<pcl::PointXYZL>::iterator it;//modify
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
	filename << path << "\\" << getName(a);
	CreateDir(filename.str());
	/*
	std::string ss1;
	ss1 = filename.str() + "\\cloud_cluster_0.xyz";
	road.width = road.points.size();
	road.height = 1;
	road.is_dense = true;
	writer.write<pcl::PointXYZ>(ss1, road, false);
	*/

	srand((int)time(NULL));
	int temp = 0;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_noise(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_noise_xyz(new pcl::PointCloud<pcl::PointXYZ>);
	/*pcl::copyPointCloud(road, *cloud_cluster);
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
	}*/
	for (pit = labels.begin(); pit != labels.end(); ++pit)//遍历所有的label
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
		for (int j = 0; j < point_num; j++)//遍历所有的point
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
		{
			*cloud_noise = *cloud_noise + *cloud_cluster1;
			continue;
		}
		k++;
		std::stringstream ss;
		ss << filename.str() << "\\cloud_cluster_" << k << ".xyz";
		cloud_cluster1->points.push_back(single_cloud);
		pcl::copyPointCloud(*cloud_cluster1, *cloud_cluster_xyz);
		cloud_cluster_xyz->width = cloud_cluster_xyz->points.size();
		cloud_cluster_xyz->height = 1;
		cloud_cluster_xyz->is_dense = true;
		writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster_xyz, false);
		*cloud_cluster = *cloud_cluster + *cloud_cluster1;
	}
	std::stringstream ss_noise;
	ss_noise << filename.str() << "\\cloud_cluster_" << ++k << ".xyz";
	pcl::copyPointCloud(*cloud_noise, *cloud_noise_xyz);
	writer.write<pcl::PointXYZ>(ss_noise.str(), *cloud_noise_xyz, false);
	/*噪音颜色红色*/
	for (int m = 0; m < cloud_noise->points.size(); m++)
	{
		cloud_noise->points[m].r = 255;
		cloud_noise->points[m].g = 0;
		cloud_noise->points[m].b = 0;
	}
	*cloud_cluster = *cloud_cluster + *cloud_noise;
	cloud_cluster->width = cloud_cluster->points.size();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;
	writer.write<pcl::PointXYZRGB>(filename.str() + "\\" + getName(a) + ".pcd", *cloud_cluster, false);
	//std::cerr << ">> Done: " << tt.toc() << " ms\n";
}