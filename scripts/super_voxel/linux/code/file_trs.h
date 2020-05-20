#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/console/time.h>
#include <conio.h>
#include <ctime>
#include <vector>
#include <string>
#include <sstream>
#include <direct.h>
using namespace std;
/*获取文件夹下的文件名
path: 文件上一级目录;
files: 带路径的文件名;
filesname: 文件名（带扩展名）。
*/
void getFiles(string path, vector<string>& files, vector<string>& filesname);
/*获取文件夹下的文件名（不带扩展名，不带路径）
full_name: 带路径的文件名。
*/
string getName(const char* full_name);
/*将.txt原始数据转化为.xyz文件
  file_path: .txt文件上一级目录。如"D:\\文档\\自动化所\\forsave_xyz";
  result_path: 转化后.xyz文件上一级目录。如""D:\\文档\\自动化所\\xyz\\"。
*/
int file_xyz(string file_path, string result_path);
/*在路径dir下创建文件夹*/
void CreateDir(string dir);
