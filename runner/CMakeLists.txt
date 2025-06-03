cmake_minimum_required(VERSION 3.0.2)
project(runner)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
        rospy
        std_msgs
        sensor_msgs
        cv_bridge
        nodelet
        pluginlib
        image_transport
)

find_package(dynamic_reconfigure REQUIRED)

generate_dynamic_reconfigure_options(
        cfg/ExtractPP.cfg
        cfg/FoamStitch.cfg
)

catkin_package(
        CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge nodelet pluginlib image_transport
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_library(extract_pp_nodelet
        src/extract_pp_nodelet.cpp
)
target_link_libraries(extract_pp_nodelet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

add_library(foam_stitch_nodelet src/foam_stitch_nodelet.cpp)
target_link_libraries(foam_stitch_nodelet ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

# 安装可执行脚本（可以运行的主程序）
catkin_install_python(PROGRAMS
        scripts/photo_capture.py
        scripts/photo_stick.py
        scripts/model_evaluation.py
        scripts/try/pure/extract_pp_nodelet.py
        scripts/try/pure/ros_high_undistort_extract_pp.py
        scripts/test1.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装 nodelet 的 Python 文件（不是直接执行，而是供 import）
install(DIRECTORY src/runner
        DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
        FILES_MATCHING PATTERN "*.py")

# 安装 launch 文件和 config
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
