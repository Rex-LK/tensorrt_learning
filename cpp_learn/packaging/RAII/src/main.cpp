#include"infer.hpp"

using namespace std;


//RAII模式下的多线程

int main(){

    auto infer = creat_infer("a");
    if(infer==nullptr){
        printf("faild.\n");
        return -1;
    }
    auto fa = infer->forward("A");
    auto fb = infer->forward("B");
    auto fc = infer->forward("C");

    printf("%s \n",fa.get().c_str());
    printf("%s \n",fb.get().c_str());
    printf("%s \n",fc.get().c_str());
    
    return 0;
}

