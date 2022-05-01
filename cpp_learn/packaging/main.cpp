#include<stdio.h>
#include<thread>
#include<queue>
#include<mutex>
#include<string>
#include<chrono>
#include<iostream>
//条件变量的头文件
#include<condition_variable>
#include<future>


using namespace std;
const int limit_ = 5;
struct Job{
    shared_ptr<promise<string>>pro;
    string input;
};
queue<Job> qjobs_;

mutex lock_;

condition_variable cv_;


//共享资源访问问题，因为queue等stl 不是线程安全的
//如果同时sleep 1s 则不会出现问题，如果生产者的频率高，则会出现队列堆积的问题，占用大量内存


//生产者
void video_capture(){
    
    int picId = 0;
    while(true){
        Job job;
        // 大括号为作用域
        {   
            //对作用域加锁,除了作用域之后自动释放
            // lock_guard<mutex>l(lock_);
            unique_lock<mutex>l(lock_);
            char name[100];
            sprintf(name,"PIC-%d",picId++);
            printf("生产了一个新图片: %s, qjobs_.size() = %d \n",name,qjobs_.size());
            
            //目标是达到这样的效果
            // if(qjobs_.size() > limit){
            //     wait();
            // }

​            //实现是这样的
​            //wait 的流程是 一旦进入wait 则解锁
​            // 一旦退出wait,则继续加锁
​            cv_.wait(l,[&](){
​                //return false 表示继续等待
​                // return true 表示不等待,跳出wait
​                return qjobs_.size() < limit_;
​            });

​            // 如何通知到wait, 及时退出。
​            // 如果队列满了，则不生产，等有空间再生产
​            // 通知问题，如何通知wait,让他及时退出
​            job.pro.reset(new promise<string>());
​            job.input = name;
​            qjobs_.push(job);

​            //假定队列有一个最大值,如果队列满了,则不进行生产,等待队列有空间再生产

​            //拿到推理结果之后，跟推理之前的图片画框，然后走下面的流程 
​            
​            //如果直接在这里进行多个模型的推理，则会变成串行模式
​            //同步模式
​            //detection -> infer
​            //face -> infer
​            //feature -> infer

​            //异步模式,因为push是很快的,push之后供其他线程进行调用
​            //detection -> push
​            //face -> push
​            //feature -> push

​            //一次进行三个模块的回收


​            //等待这个job处理完毕，拿结果 //阻塞等待


​            //拿到推理结果，跟推理之前的图像一起进行画框，然后走下面的流程


​        }

​        auto result =  job.pro->get_future().get();
​        printf("JOB %s -> %s \n",job.input.c_str(),result.c_str());
​        
​        this_thread::sleep_for(chrono::milliseconds(500));
​    }
}

// 消费者
void infer_work(){
    while(true){
        if(!qjobs_.empty()){
            {
                lock_guard<mutex>l(lock_);
                auto pjob = qjobs_.front();
                qjobs_.pop();

​                //消费掉一个,就可以通知wait,去跳出等待
​                cv_.notify_one();
​                printf("消费掉一个图片: %s\n",pjob.input.c_str());

​                auto result = pjob.input + " ---- infer";
​                pjob.pro->set_value(result);
​            }
​            
​            this_thread::sleep_for(chrono::milliseconds(1000));

​        }
​        this_thread::yield();   
​    }
}

int main(){
    thread t0(video_capture);
    thread t1(infer_work);
    t0.join();
    t1.join();
    return 0;
}