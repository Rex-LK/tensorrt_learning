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
        thread_ = thread(boost::bind(&ClientNodelet::run, this));
    }

    void ClientNodelet::run()
    {

        setlocale(LC_ALL, "");
        ros::service::waitForService("yolov5");
        ros::ServiceClient client = nh_.serviceClient<base::RosImage>("yolov5");

        Mat image = cv::imread("/home/rex/Documents/yolo5-6.0-ros/src/yolov5-6.0/images/bus.jpg");
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
