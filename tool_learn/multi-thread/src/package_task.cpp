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
    //等待主线程的p_in 设置号值之后再继续
    int ret_b = b * 2;
    return ret_a+ret_b;
}

//为了简化上述方法，可以使用async

int main(){

    //使用async可以在线程中获得返回值
    //async一点创建新的线程做计算，如果使用launch::async则会开启新的线程
    //launch::deferred 为延迟调用,当有fu.get()时，才会开启线程
    future<int> fu = async(launch::async,task,1,2);
    cout <<"return ret is :" << fu.get() <<endl;

}