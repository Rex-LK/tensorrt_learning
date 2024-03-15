
#include "infer.hpp"

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace fastdet;
using namespace cv;

struct Job
{
    shared_ptr<promise<vector<TargetBox>>> pro;
    Mat input;
};

class InferImpl : public Infer
{
public:
    virtual ~InferImpl() { stop(); }

    void stop();
    // 模型路径
    bool startup(const string &param_path, const string &model_path);

    virtual shared_future<vector<TargetBox>> commit(Mat &input) override;

    void worker(promise<bool> &pro);

private:
    atomic<bool> running_{false};
    string file_;
    thread worker_thread_;
    queue<Job> jobs_;
    mutex lock_;
    condition_variable cv_;
    shared_ptr<FastDet*> fast_det_ = nullptr;
    string param_path_;
    string model_path_;
    int batch_size = 1;
    int input_width_ = 352;
    int input_height_ = 352;
    string input_name_ = "input.1";
    string output_name_ = "758";
    int infer_thread_ = 6;
    int class_num = 80;
};

void InferImpl::stop()
{
    if (running_)
    {
        running_ = false;
        cv_.notify_one();
    }

    if (worker_thread_.joinable())
        worker_thread_.join();
}

bool InferImpl::startup(const string &param_path, const string &model_path)
{
    param_path_ = param_path;
    model_path_ = model_path;
    running_ = true; // 启动后，运行状态设置为true

    // 线程传递promise的目的，是获得线程是否初始化成功的状态
    // 而在线程内做初始化，好处是，初始化跟释放在同一个线程内
    // 代码可读性好，资源管理方便
    promise<bool> pro;
    worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));
    /*
        注意：这里thread 一构建好后，worker函数就开始执行了
        第一个参数是该线程要执行的worker函数，第二个参数是this指的是class
       InferImpl，第三个参数指的是传引用，因为我们在worker函数里要修改pro。
     */
    return pro.get_future().get();
}

shared_future<vector<TargetBox>> InferImpl::commit(Mat &input)
{
    Job job;
    job.input = input;
    job.pro.reset(new promise<vector<TargetBox>>());

    shared_future<vector<TargetBox>> fut =
        job.pro->get_future(); // 将fut与job关联起来
    {
        lock_guard<mutex> l(lock_);
        jobs_.emplace(std::move(job));
    }
    cv_.notify_one(); // 通知线程进行推理
    return fut;
}

void InferImpl::worker(promise<bool> &pro)
{
    // load model
    // 加载模型
    fast_det_.reset(new FastDet(input_width_, input_height_, param_path_, model_path_));
    if (fast_det_ == nullptr)
    {
        // failed
        pro.set_value(false);
        printf("Load model failed: %s\n", file_.c_str());
        return;
    }

    // load success
    pro.set_value(true); // 这里的promise用来负责确认infer初始化成功了

    vector<Job> fetched_jobs;
    while (running_)
    {
        {
            unique_lock<mutex> l(lock_);
            cv_.wait(l, [&]()
                     { return !running_ || !jobs_.empty(); }); // 一直等着，cv_.wait(lock, predicate) // 如果 running不在运行状态
                                                               // 或者说 jobs_有东西 而且接收到了notify one的信号

            if (!running_)
                break; // 如果 不在运行 就直接结束循环

            for (int i = 0; i < batch_size && !jobs_.empty();
                 ++i)
            { // jobs_不为空的时候
                fetched_jobs.emplace_back(
                    std::move(jobs_.front())); // 就往里面fetched_jobs里塞东西
                jobs_
                    .pop(); // fetched_jobs塞进来一个，jobs_那边就要pop掉一个。（因为move）
            }
        }

        // 一次加载一批，并进行批处理
        // forward(fetched_jobs)
        for (auto &job : fetched_jobs)
        {
            int img_width = job.input.cols;
            int img_height = job.input.rows;
            fast_det_->prepare_input(job.input);
            fast_det_->infrence(input_name_, output_name_, infer_thread_);
            fast_det_->postprocess(img_width, img_height, class_num, 0.65);
            job.pro->set_value(fast_det_->nms_boxes);
        }
        fetched_jobs.clear();
    }
    printf("Infer worker done.\n");
}

shared_ptr<Infer> create_infer(const std::string &param_path,
                               const std::string &model_path)
{
    shared_ptr<InferImpl> instance(
        new InferImpl()); // 实例化一个推理器的实现类（inferImpl），以指针形式返回
    // 线程中是否加载好模型
    if (!instance->startup(
            param_path,
            model_path))
    {                     // 推理器实现类实例(instance)启动。这里的file是engine
                          // file
        instance.reset(); // 如果启动不成功就reset
    }
    return instance;
}
