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


void task(int a, future<int>&b,  promise<int>& ret){
    int ret_a = a * a;
    //等待主线程的p_in 设置号值之后再继续
    int ret_b = b.get()* 2;

    ret.set_value(ret_a+ret_b);

}

//有时候 在给线程传值是,可能不需要立马传递值,可能需要过一段时间传递值

int main(){
    int ret = 0;

    promise<int>p_ret;
    future<int>f_ret = p_ret.get_future();

    promise<int>p_in;
    future<int>f_in = p_in.get_future();

    thread t(task,1,ref(f_in),ref(p_ret));
    
    //do

    p_in.set_value(8);
     
    //

    cout <<"return ret is :" << f_ret.get() <<endl;
    if(t.joinable())
        t.join();

}