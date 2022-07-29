#ifndef CLIENT_CLIENT_NODELET_H
#define CLIENT_CLIENT_NODELET_H

#include <std_msgs/String.h>
#include <sys/stat.h>
#include <string>
#include <thread>

#include <string>
#include <vector>
#include "base/base.h"
#include "base/RosImage.h"
using namespace std;

namespace client_nodelet
{
    class ClientNodelet : public base::Base
    {
    public:
        ~ClientNodelet();

        virtual void onInit();

    private:
        void run();

    private:
        thread thread_;
    };
}

#endif