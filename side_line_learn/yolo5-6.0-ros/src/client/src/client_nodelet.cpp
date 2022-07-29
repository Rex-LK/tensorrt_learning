#include <iostream>

#include "client_nodelet/client_nodelet.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace client_nodelet
{
    ClientNodelet::~ClientNodelet() {}

    void ClientNodelet::onInit()
    {
        init();
        image_path = params_["image_path"];
        client_name = params_["client_name"];
        thread_ = thread(boost::bind(&ClientNodelet::run, this));
    }

    void ClientNodelet::run()
    {

        setlocale(LC_ALL, "");
        ros::service::waitForService(client_name);
        ros::ServiceClient client = nh_.serviceClient<base::RosImage>(client_name);

        Mat image = cv::imread(image_path);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        sensor_msgs::Image msg1 = *msg;
    
        base::RosImage ai;
        ai.request.image = *msg;
        bool flag = client.call(ai);
        if (flag)
        {
            ROS_INFO("detected target,result = %s", ai.response.result.c_str());
        }
        else
        {
            ROS_INFO("failed,no target");
        }
        ros::Duration(0.2).sleep();
        
    }

} // namespace server_nodelet
PLUGINLIB_EXPORT_CLASS(client_nodelet::ClientNodelet,
                       nodelet::Nodelet)
