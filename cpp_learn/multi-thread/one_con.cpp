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

condition_variable cv;

mutex mtx;
deque<int>q;


void task1(){
    int i=0;
    while(true){
        unique_lock<mutex> lock(mtx);
        q.emplace_back(i);
        
        cv.notify_one();
        // cv.notify_all();

        //每次生产出一个数据后，就去唤醒 cv_wait
        if(i<999999){
            i++;
        }
        else{
            i=0;
        }
    }
}

void task2(){
    int data = 0;
    while(true){
        unique_lock<mutex>lock(mtx);

        //如果队列为空，就利用wait进入休眠状态,同时释放锁,被唤醒时加锁
        if(q.empty()){
            cv.wait(lock);
        }

        if(!q.empty()){
            data = q.front();
            q.pop_front();
            cout<<"Get value from que:"<<data<<endl;
        }
    }
}



// void task3(){
//     int data = 0;
//     while(true){
//         unique_lock<mutex>lock(mtx);

//         //如果队列为空，就利用wait进入休眠状态,同时释放锁,被唤醒时加锁
//         if(q.empty()){
//             cv.wait(lock);
//         }

//         if(!q.empty()){
//             data = q.front();
//             q.pop_front();
//             cout<<"Get value from que:"<<data<<endl;
//         }
//     }
// }

int main(){

    thread t1(task1);
    thread t2(task2);
    // thread t3(task3);
    if(t1.joinable())
        t1.join();
    if(t2.joinable())
        t2.join();
    // if(t3.joinable())
    //     t3.join();
    return 0;
}