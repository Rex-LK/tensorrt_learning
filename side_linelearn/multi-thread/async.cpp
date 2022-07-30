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


int task(int a,int b){
    int ret_a = a * a;
    int ret_b = b * 2;
    return ret_a+ret_b;
}


int main(){


    //对线程进行包装,之后传递参数
    // packaged_task<int(int,int)> t(task);
    //启动线程
    // t(1,2);

    //包装时传递参数

    packaged_task<int()> t(bind(task,1,2));
    t();

    //获取返回值
    int res = t.get_future().get();

    cout <<"res:"<<res<<endl;
}