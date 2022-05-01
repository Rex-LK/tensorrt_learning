#include "infer.hpp"
#include<thread>
#include<queue>
#include<mutex>
#include<future>

using namespace std;
//RAII + 接口模式I
//常用模式
//异常逻辑需要耗费大量时间
//异常逻辑如果没写好,或者没写，就会造成封装的不安全性，导致程序崩溃且复杂性变高

struct Job
{
    shared_ptr<promise<string>>pro;
    string input;
};

class InferImpl :public InferInterface{
public:
    virtual ~InferImpl(){
        work_running_ = false;
        cv_.notify_one();

        if(worker_thread_.joinable()){
            worker_thread_.join();
        }
    }
    
    bool load_model(const string &file){
        
        // 尽量使得资源哪里使用，然后在哪里释放
        //线程内传回返回值的问题 使用future
        promise<bool> pro;
        work_running_ = true;
        worker_thread_ = thread(&InferImpl::worker,this,file,std::ref(pro));
        return pro.get_future().get();
    }

    virtual shared_future<string> forward(string pic) override{
        //往队列抛任务
        Job job;
        job.pro.reset(new promise<string>); 
        job.input = pic;
        //资源访问加锁
        lock_guard<mutex>l(job_lock_);
        qjobs_.push(job);
        
        // 被动通知,一旦有新的任务出现,立即通知线程进行预测
        // 唤醒子线程
        cv_.notify_one();


        //这里写.get会进行阻塞，影响性能
        // return job.pro->get_future().get();
         return job.pro->get_future();


    }
 
    
    //实际模型推理的部分
    void worker(string file, promise<bool>& pro){
        //实现模型的加载、使用以及释放
        string context = file;
        if(context.empty()){
            pro.set_value(false);
        }
        else{
            pro.set_value(true);
        }

        int max_batchsize = 5;
        vector<Job> jobs;

        int batch_id = 0;

        while(work_running_){
            //等待被唤醒
            //往队列取任务并执行的过程
            unique_lock<mutex>l(job_lock_);
            cv_.wait(l,[&](){
                //true 退出等待
                //false 继续等待
                //根据任务队列是否为空进行判断 或者work_running_信号
                return !qjobs_.empty() || !work_running_;
            });

            if(!work_running_){
                //程序因为终止信号而推出wait
                break;
            }

            //可以一次拿一批数据，最大拿maxbatcsize个
            while(jobs.size() < max_batchsize && !qjobs_.empty()){
                jobs.emplace_back(qjobs_.front());
                qjobs_.pop();
            }
            //do inference,如果队列一直为空,则这里会一直循环，于是可以采用条件变量来唤醒

            //执行batch_推理
            for(int i = 0;i<jobs.size();i++){
                auto&job = jobs[i];
                char result[100];
                sprintf(result,"%s:batch->%d[%d]",job.input.c_str(),batch_id,jobs.size());
                job.pro->set_value(result);
            }
            batch_id++;
            jobs.clear();
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
        
        printf("释放:%s\n",context.c_str());
        context.clear();
        printf("work done.\n");
    }

private:
    //线程的运行状态
    atomic<bool> work_running_{false};
    //代表tensorrt中的上下文
    thread worker_thread_;
    //任务队列
    queue<Job> qjobs_;
    mutex job_lock_;
    condition_variable cv_;
}; 

//改进
//RAII
//如果获取infer实例即加载模型，加载模型失败，则获取资源失败，强绑定
//加载资源成功，则资源获取成功

//避免外部load_model,封装后,外部只需要查看laod_model为true或者false
//一个实例的 load_model不会超过一次

//接口模式
//1.解决load_model还能被外部 调用的问题
//2.结局成员变量对外课件的问题
//  对于成员函数是特殊类型的，比如说是cudaString_t,那么使用者毕竟会包含cuda_runtime.h,
//  造成没必要的包含头文件
shared_ptr<InferInterface> creat_infer(const string&file){
    shared_ptr<InferImpl> instance(new InferImpl);
    if(!instance->load_model(file)){
        instance.reset();
    }
    return instance;
}