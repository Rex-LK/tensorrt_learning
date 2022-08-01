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


//生产者线程
void prod(){
    int i=0;
    while(true){
        unique_lock<mutex> lock(mtx);
        q.emplace_back(i);
        
        //随机唤醒某一个线程
        cv.notify_one();

        // 去唤醒所有线程
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

//消费者线程1
void con1(){
    int data = 0;
    while(true){
        unique_lock<mutex>lock(mtx);

        //如果队列为空，就利用wait进入休眠状态,同时释放锁,被唤醒时加锁
        if(q.empty()){
            //可能有多个线程调用了这个cv_wait，所以会导致报错
            cv.wait(lock);
            //含义如下，
            //lock.unlock()
                //cv.wait()
        }

        data = q.front();
        q.pop_front();
        cout<<"Get value from que:"<<data<<endl;
    }
}


//消费者线程2
//如果此时增加了一个消费者线程,将会出现问题

//虚假唤醒问题  即队列为空，但是消费者线程却被唤醒了，去取队列里面的值，就报错了、
//发生错误的情形如下:
//在线程1中，此时队列q中有一个值，线程1正常pop，然后打印，接着进入到下一个循环，运行到 unique_lock<mutex>lock(mtx);时
//此时队列 元素+1,然后随机的唤醒了线程2,但此时线程1拥有队列的锁，于是线程1拿到了队列中的值，然后进行的pop，此时队列为空，
//此时线程2再去拿队列中的值，就会发生异常

void con2(){
    int data = 0;
    while(true){
        unique_lock<mutex>lock(mtx);

        //如果队列为空，就利用wait进入休眠状态,同时释放锁,被唤醒时加锁
        if(q.empty()){
            cv.wait(lock);
        }
        data = q.front();
        q.pop_front();
        cout<<"Get value from que:"<<data<<endl;
    }
}

int main(){

    thread t1(prod);
    thread t2(con1);
    thread t3(con2);
    if(t1.joinable())
        t1.join();
    if(t2.joinable())
        t2.join();
    if(t3.joinable())
        t3.join();
    return 0;
}