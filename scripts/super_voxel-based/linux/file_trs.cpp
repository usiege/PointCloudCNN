#include "file_trs.h"
#include <dirent.h>
#include<sys/types.h>
#include <unistd.h>

/*????????????????
path: ??????????;
files: ??¡¤?????????;
filesname: ??????????????????
*/
void getFiles(string path, vector<string>& files, vector<string>& filesname)
{
	//??????
	long   hFile = 0;
	struct dirent *ptr;
	DIR *dir;
	dir=opendir(path.c_str());

	cout << "In getFiles: " << endl;
	while((ptr=readdir(dir))!=NULL)
	{

		//????'.'??'..'??????
		if(ptr->d_name[0] == '.')
			continue;
		string fullname=path+"/"+ptr->d_name;

		files.push_back(fullname);
		string p=ptr->d_name;

		int pos = p.find_last_of('.');
		cout<<fullname<<endl;

		filesname.push_back(p.substr(pos-9, 9));
		cout << p.substr(pos-9, 9) << endl;

	}
	cout << "away from getFiles!" << endl;
	closedir(dir);
}


/*?????????????????????????????????¡¤????
full_name: ??¡¤???????????
*/
string getName(const char* full_name)
{
	//cout << "full name: " << full_name << endl;

	string file_name = full_name;
	const char*  mn_first = full_name;
	const char*  mn_last = full_name + strlen(full_name);
	if (strrchr(full_name, '/') != NULL)
		mn_first = strrchr(full_name, '/') + 1;
	else if (strrchr(full_name, '/') != NULL)
		mn_first = strrchr(full_name, '/') + 1;
	if (strrchr(full_name, '.') != NULL)
		mn_last = strrchr(full_name, '.');
	if (mn_last < mn_first)
		mn_last = full_name + strlen(full_name);

	file_name.assign(mn_first, mn_last);
	//cout << "file name: " << file_name << endl;

	return file_name;
}
/*??.txt??????????.xyz???
file_path: .txt??????????????"D:\\???\\???????\\???????\\forsave_xyz";
result_path: ?????.xyz??????????????"D:\\???\\???????\\???????\\xyz"??
*/
int file_xyz(string file_path, string result_path)
{
	vector<string> files;
	vector<string> filename, filename1;
	getFiles(file_path, files, filename);
	for (vector<string>::iterator it = files.begin(), pit = filename.begin();
	it < files.end(), pit<filename.end(); it++, pit++)
	{
		fstream fin(*it, ios::in);
		ifstream infile;
		ostringstream str;
		char a[100];
		strcpy(a, (*pit).c_str());

		str << result_path <<"/"<< getName(a) << ".xyz";
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
	}
	cout<<"OK!"<<endl;
	return 0;
}
void CreateDir(string dir)
{
	int m = 0, n;
	string str1, str2;

	str1 = dir;
	//str2 = str1.substr(0, 2);
	//str1 = str1.substr(3, str1.size());

	//m = str1.find('/');
	//str1 = str1.substr(m, str1.size());

	while (m >= 0)
	{
		m = str1.find('/');
		//cout<<"m="<<m<<endl;
		str2 += '/'+str1.substr(0, m);
		//str2 += str1.substr(0, m);

		//cout<<"str2="<<str2<<endl;
		//n = _access(str2.c_str(), 0); //?§Ø??????????
		n = access(str2.c_str(), 0); //?§Ø??????????
		if (n == -1)
		{
			//_mkdir(str2.c_str());     //??????
			mkdir(str2.c_str(),0777);     //??????
		}

		str1 = str1.substr(m+1, str1.size());
	}
}
