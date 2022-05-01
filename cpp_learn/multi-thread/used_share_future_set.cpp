
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


void task(int a,shared_future<int>&b,  promise<int>& ret){
    int ret_a = a * a;
    //等待主线程的p_in 设置号值之后再继续
    int ret_b = b.get()* 2;

    ret.set_value(ret_a+ret_b);

}


int main(){
    int ret = 0;
    //p_ret是不能赋值的
    promise<int>p_ret;
    //可以通过move实现
    promise<int>p_ret2 = move(p_ret);

    promise<int>p_ret;
    future<int>f_ret = p_ret.get_future();

    promise<int>p_in;
    future<int>f_in = p_in.get_future();

    //当创建了很多线程时
    //由于线程中 b.get()只能用一次,这里会出现异常
    //解决办法
    shared_future<int> s_f = f_in.share();

    thread t(task,1(s_f,ref(p_ret));
    thread t(task,1(s_f,ref(p_ret));
    thread t(task,1(s_f,ref(p_ret));
    thread t(task,1(s_f,ref(p_ret));

    p_in.set_value(8);

    cout <<"return ret is :" << f_ret.get() <<endl;
    if(t.joinable())
        t.join();

}