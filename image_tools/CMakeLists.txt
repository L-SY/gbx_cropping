cmake_minimum_required(VERSION 3.0.2)
project(image_tools)

## 找到依赖的包
find_package(catkin REQUIRED COMPONENTS
        rospy
        sensor_msgs
        cv_bridge
)

## catkin 包配置
catkin_package()

## 包含头文件目录（如果有 C++ 代码）
include_directories(
        ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
        src/image_saver.py
        src/image_cropping.py
        src/image_cropping.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)