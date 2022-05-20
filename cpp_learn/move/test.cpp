#include <iostream>
using namespace std;

template<typename T>

void func(T& val)
{
    cout <<"right-value" << val << endl;
}
template<typename T>
void func(T&& val)
{
    cout <<"right-value" << val << endl;
}

int main()
{   
    //demo1
    int year = 2020;
    func(year); //传入左值  引用折叠
    func(2020); //传入右值  引用折叠
    return 0;
}
