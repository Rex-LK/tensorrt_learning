#include<thread>
#include<queue>
#include<mutex>
#include<string>
#include<chrono>
#include<condition_variable>
#include<memory>
#include<future>
#include<stdio.h>

using namespace std;


//RAII + 接口模式I
//常用模式
//异常逻辑需要耗费大量时间
//异常逻辑如果没写好,或者没写，就会造成封装的不安全性，导致程序崩溃且复杂性变高
class Infer{
public:
    bool load_model(const string &file){
        
        if(!context_.empty()){
            destory();
        }
        //加载模型 
        context_ = file;
        return true;
    }

    void forward(){

        //异常逻辑
        if(context_.empty()){
            //说明模型没有加载上
            //没有对异常处理做处理
            printf("模型没有加载成功\n");
            return;
        }
        printf("使用%s进行推理 \n",context_.c_str());
    }

    void destory(){
        //context_.clear();
    }

private:
    //代表tensorrt中的上下文
    string context_;
}; 



int main(){

    //直接进行推理
    Infer infer;
    infer.load_model("a");
    infer.forward();
    return 0;
}

