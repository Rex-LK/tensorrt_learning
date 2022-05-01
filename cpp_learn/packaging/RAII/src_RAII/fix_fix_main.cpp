#include"infer.hpp"

using namespace std;


int main(){

    auto infer = creat_infer("a");
    if(infer==nullptr){
        printf("faild.\n");
        return -1;
    }
    infer->forward();
    return 0;
}

