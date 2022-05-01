#ifndef INFER_HPP
#define INFER_HPP

#include<memory>
#include<string>

// 总结原则
// 1、头文件尽量只包含需要的部分,因为这个头文件可能被多个人进行调用
// 2、外界不需要的，尽量不需要让他看到，保证定义的简洁
// 3、不要在头文件写 using namespace  但可以在cpp中写

//RAII + 接口模式

class InferInterface{
public:
    virtual void forward() = 0;
};
std::shared_ptr<InferInterface> creat_infer(const std::string&file);

#endif./ INFER_HPP