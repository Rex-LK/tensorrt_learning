#include<iostream>
#include<stdio.h>
#include <utility>
#include<cstring>
using namespace std;
class String
{
public:
    String(const char* buf,const char* buf2)
    {
    	_buf = new char[strlen(buf) + 1];
        strcpy(_buf, buf);	// 内存拷贝
        _buf2 = new char(strlen(buf2) + 1);
        strcpy(_buf2, buf2);	// 内存拷贝
        cout<<"使用普通构造函数" <<endl;

    }
    
	String(const String& str)
    {
     	_buf = new char[strlen(str._buf) + 1];
        strcpy(_buf, str._buf);	// 内存拷贝

        _buf2 = new char[strlen(str._buf2) + 1];
        strcpy(_buf2, str._buf2);	// 内存拷贝
        cout<<"使用构造函数" <<endl;
    }
    
    String(String&& str)
    {
        _buf = str._buf;		// 直接使用内存地址
        str._buf = nullptr;
        _buf2 = str._buf2;		// 直接使用内存地址
        str._buf2 = nullptr;
        cout<<"移动构造函数" <<endl;

    }
    
private:
	char* _buf;
    char* _buf2;		
};

int main()
{
    String str("hello world!","dd");
    String str1(str);
    //把str里面的东西拿出来 这个东西叫做右值，指向这个右值的叫 右值引用
    String str2(std::move(str));
    return 0;
}
