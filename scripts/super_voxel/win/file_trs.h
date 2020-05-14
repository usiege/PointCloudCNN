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
/*��ȡ�ļ����µ��ļ���
path: �ļ���һ��Ŀ¼;
files: ��·�����ļ���;
filesname: �ļ���������չ������
*/
void getFiles(string path, vector<string>& files, vector<string>& filesname);
/*��ȡ�ļ����µ��ļ�����������չ��������·����
full_name: ��·�����ļ�����
*/
string getName(const char* full_name);
/*��.txtԭʼ����ת��Ϊ.xyz�ļ�
  file_path: .txt�ļ���һ��Ŀ¼����"D:\\�ĵ�\\�Զ�����\\forsave_xyz";
  result_path: ת����.xyz�ļ���һ��Ŀ¼����""D:\\�ĵ�\\�Զ�����\\xyz\\"��
*/
int file_xyz(string file_path, string result_path);
/*��·��dir�´����ļ���*/
void CreateDir(string dir);
