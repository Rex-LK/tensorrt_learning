#ifndef _BASE_H
#define _BASE_H

#include <unordered_map>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "boost/type_index.hpp"

#include "ros/ros.h"
#include <ros/package.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include "std_msgs/String.h"


namespace base
{

	class Base : public nodelet::Nodelet
	{
	protected:
		ros::NodeHandle nh_;
		ros::NodeHandle pnh_;
		std::unordered_map<std::string, std::string> params_;


	public:
		Base(){};
		~Base(){};

		void init()
		{
			nh_ = getNodeHandle();
			pnh_ = getPrivateNodeHandle();
			std::string node_name = getName();

			std::vector<std::string> param_names;
			if (pnh_.getParamNames(param_names))
			{
				for (std::string name : param_names)
				{
					std::string param_name;
					bool valid = get_param(node_name, name, param_name);
					if (valid)
					{
						std::string param_value;
						pnh_.getParam(name, param_value);
						ROS_INFO("settings: %s,%s", param_name.c_str(), param_value.c_str());
						params_[param_name] = param_value;
					}
				}
			}
		}

		bool get_param(std::string &node_name, std::string &name,
							std::string &param_name)
		{
			std::stringstream ss(name);
			bool valid = false;
			while (getline(ss, param_name, '/'))
			{
				if ("/" + param_name == node_name)
				{
					valid = true;
				}
			}
			return valid;
		}
	}; 

} 

#endif