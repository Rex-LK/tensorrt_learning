# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import stat
import sys

# find the import for catkin's python package - either from source space or from an installed underlay
if os.path.exists(os.path.join('/opt/ros/noetic/share/catkin/cmake', 'catkinConfig.cmake.in')):
    sys.path.insert(0, os.path.join('/opt/ros/noetic/share/catkin/cmake', '..', 'python'))
try:
    from catkin.environment_cache import generate_environment_script
except ImportError:
    # search for catkin package in all workspaces and prepend to path
    for workspace in '/home/rex/Desktop/tensorrt_learning/side_line_learn/yolo5-6.0-ros/devel_isolated/yolov5_infer_nodelet;/home/rex/Desktop/tensorrt_learning/side_line_learn/yolo5-6.0-ros/devel_isolated/client_nodelet;/home/rex/Desktop/tensorrt_learning/side_line_learn/yolo5-6.0-ros/devel_isolated/base;/opt/ros/noetic'.split(';'):
        python_path = os.path.join(workspace, 'lib/python3/dist-packages')
        if os.path.isdir(os.path.join(python_path, 'catkin')):
            sys.path.insert(0, python_path)
            break
    from catkin.environment_cache import generate_environment_script

code = generate_environment_script('/home/rex/Desktop/tensorrt_learning/side_line_learn/yolo5-6.0-ros/devel_isolated/yolov5_server_nodelet/env.sh')

output_filename = '/home/rex/Desktop/tensorrt_learning/side_line_learn/yolo5-6.0-ros/build_isolated/yolov5_server_nodelet/catkin_generated/setup_cached.sh'
with open(output_filename, 'w') as f:
    # print('Generate script for cached setup "%s"' % output_filename)
    f.write('\n'.join(code))

mode = os.stat(output_filename).st_mode
os.chmod(output_filename, mode | stat.S_IXUSR)
