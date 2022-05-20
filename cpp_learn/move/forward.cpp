#include <iostream>
#include <utility>


void Print(int& val) {
    std::cout << "lvalue refrence: val=" << val << std::endl;
}

void Print(int&& val) {
    std::cout << "rvalue refrence: val=" << val << std::endl;
}

// template<typename T>
// void TPrint(T &&t) {
//     //传进来为左值时，t为左值引用
//     //传进来右值时，t为右值引用  引用折叠
//     //按道理来说 传进来的值为右值引用时，会调用 Print(int &&val) 但实际上调用了Print(int &&val)
//     //因为传进来的 右值引用，这个右值引用实际上是左值
//     return Print(t);
// }
template<typename T>
void TPrint(T&& t) {
    //这里会根据 传进来的值判断是左值还是右值
    //然后活使用forward保持原来的属性
    //如果t为左值引用，forward会返回一个左值然后传递给Print
    //如果t为右值引用，forward会返回一个右值然后传递给Print
    //从而实现完美转发
    return Print(std::forward<T>(t));
}


// template<typename _Tp>
// constexpr _Tp&&
// forward(typename std::remove_reference<_Tp>::type&& __t) noexcept
// {
//     static_assert(!std::is_lvalue_reference<_Tp>::value, "template argument"
//         " substituting _Tp is an lvalue reference type");
        //强制类型转化
//     return static_cast<_Tp&&>(__t);
// }


//左值和右值
//左值引用和右值引用
//左值引用和右值引用 全为左值

int main() {
    int date = 1021;
    TPrint(date); // 传入左值引用
    TPrint(501);  //传入右值引用  实际上 "右值引用" 为左值 

    return 0;
}
