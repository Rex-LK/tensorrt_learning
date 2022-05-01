#include "infer.hpp"

using namespace std;
//RAII + 接口模式I
//常用模式
//异常逻辑需要耗费大量时间
//异常逻辑如果没写好,或者没写，就会造成封装的不安全性，导致程序崩溃且复杂性变高
class InferImpl :public InferInterface{
public:
    bool load_model(const string &file){
        
        //加载模型 
        context_ = file;
        return true;
    }

    void forward(){

        printf("使用%s进行推理 \n",context_.c_str());
    }
 
    void destory(){
        //context_.clear();
    }

private:
    //代表tensorrt中的上下文
    string context_;
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