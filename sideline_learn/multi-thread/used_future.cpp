#include<stdio.h>
#include<thread>
#include<queue>
#include<mutex>
#include<string>
#include<chrono>
#include<iostream>
#include<condition_variable>
#include<future>

using namespace std;


void task(int a,int b,  promise<int>& ret){
    int ret_a = a * a;
    int ret_b = b * 2;

    ret.set_value(ret_a+ret_b);

}

//这个版本导致代码太多,使用future可以优雅的解决

int main(){
    int ret = 0;

    promise<int>p;
    future<int>f = p.get_future(); //将f与p关联起来
    
    thread t(task,1,2,ref(p));

    //f.get()会阻塞在这里等待 p给f传递值，且f.get()只能用一次
    cout <<"return ret is :" << f.get() <<endl;
    if(t.joinable())
        t.join();

}