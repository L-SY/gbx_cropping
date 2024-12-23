cmake_minimum_required(VERSION 3.0.2)
project(gbx_cropping)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        nodelet
        sensor_msgs
        cv_bridge
        image_transport
        std_msgs
)

find_package(OpenCV REQUIRED)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES gbx_cropping_nodelet
        CATKIN_DEPENDS roscpp nodelet sensor_msgs cv_bridge image_transport std_msgs
        DEPENDS OpenCV
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_library(gbx_cropping_nodelet
        src/gbx_cropping.cpp
)

target_link_libraries(gbx_cropping_nodelet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

install(TARGETS gbx_cropping_nodelet
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(FILES gbx_cropping.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)