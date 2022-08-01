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

mutex mtx;
condition_variable cv;

void task(int a,int b, promise<int>& ret){
    int ret_a = a * a;
    int ret_b = b * 2;
    ret = ret_a + ret_b;
    cv.notify_one();
}

//这个版本导致代码太多,使用future可以优雅的解决

int main(){
    int ret = 0;
    //如果需要传递引用，则需要加ref

    //将p传递进去
    thread t(task,1,2,ref(ret));

    unique_lock<mutex>lock(mtx);
    //等待线程通知，然后往下执行
    cv.wait(lock);
    cout<< "return value is" << ret<<endl;
    if(t.joinable())
        t.join();

}