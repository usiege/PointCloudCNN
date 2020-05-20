#include <file_trs.h>
/*获取文件夹下的文件名
path: 文件上一级目录;
files: 带路径的文件名;
filesname: 文件名（带扩展名）。
*/
void getFiles(string path, vector<string>& files, vector<string>& filesname)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("/").append(fileinfo.name), files, filesname);
			}
			else
			{
				files.push_back(p.assign(path).append("/").append(fileinfo.name));
				filesname.push_back(fileinfo.name);
				//files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
/*获取文件夹下的文件名（不带扩展名，不带路径）
full_name: 带路径的文件名。
*/
string getName(const char* full_name)
{
	string file_name = full_name;
	const char*  mn_first = full_name;
	const char*  mn_last = full_name + strlen(full_name);
	if (strrchr(full_name, '\\') != NULL)
		mn_first = strrchr(full_name, '\\') + 1;
	else if (strrchr(full_name, '/') != NULL)
		mn_first = strrchr(full_name, '/') + 1;
	if (strrchr(full_name, '.') != NULL)
		mn_last = strrchr(full_name, '.');
	if (mn_last < mn_first)
		mn_last = full_name + strlen(full_name);

	file_name.assign(mn_first, mn_last);

	return file_name;
}
/*将.txt原始数据转化为.xyz文件
file_path: .txt文件上一级目录。如"D:\\文档\\自动化所\\forsave_xyz";
result_path: 转化后.xyz文件上一级目录。如"D:\\文档\\自动化所\\xyz"。
*/
int file_xyz(string file_path, string result_path)
{
	vector<string> files;
	vector<string> filename, filename1;
	getFiles(file_path, files, filename);
	for (vector<string>::iterator it = files.begin(), pit = filename.begin(); it < files.end(), pit<filename.end(); it++, pit++)
	{
		/*
		fstream fin(*it, ios::in);
		ifstream infile;
		ostringstream str;
		char a[100];
		strcpy(a, (*pit).c_str());
		str << result_path <<"\\"<< getName(a) << ".xyz";
		char txt[200];
		infile.open(*it);
		ofstream fout(str.str());
		if (!fin)
		{
			cerr << "can not open file" << endl;
			return -1;
		}
		char c;
		int lineCnt = 0;
		while (fin.get(c))
		{
			if (c == '\n')
				lineCnt++;
		}
		fout << "# .PCD v0.7 - Point Cloud Data file format\n"
			<< "VERSION 0.7\n"
			<< "FIELDS x y z rgb\n"
			<< "SIZE 4 4 4 4\n"
			<< "TYPE F F F F\n"
			<< "COUNT 1 1 1 1\n"
			<< "WIDTH " << lineCnt << "\n"
			<< "HEIGHT 1\n"
			<< "VIEWPOINT 0 0 0 1 0 0 0\n"
			<< "POINTS " << lineCnt << "\n"
			<< "DATA ascii\n";
		while (!infile.eof())
		{
			infile.getline(txt, 100);
			fout << txt << "\n";
		}
		fin.close();
		fin.clear();
		fout.close();
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::io::loadPCDFile(str.str(), *cloud_tmp);
		pcl::copyPointCloud(*cloud_tmp, *cloud);
		pcl::PCDWriter writer;
		writer.write(str.str(),*cloud, false);
		*/
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPCDFile(*it, *cloud);
		pcl::PCDWriter writer;
		ostringstream str;
		char a[100];
		strcpy(a, (*pit).c_str());
		str << result_path << "\\" << getName(a) << ".xyz";
		writer.write(str.str(), *cloud, false);
	}
	return 0;
}
void CreateDir(string dir)
{
	int m = 0, n;
	string str1, str2;

	str1 = dir;
	str2 = str1.substr(0, 2);
	str1 = str1.substr(3, str1.size());

	while (m >= 0)
	{
		m = str1.find('\\');

		str2 += '\\' + str1.substr(0, m);
		n = _access(str2.c_str(), 0); //判断该目录是否存在
		if (n == -1)
		{
			_mkdir(str2.c_str());     //创建目录
		}

		str1 = str1.substr(m + 1, str1.size());
	}
}