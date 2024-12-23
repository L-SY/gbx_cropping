cmake_minimum_required(VERSION 3.5)
project(hk_camera)

set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        nodelet
        message_generation
        dynamic_reconfigure
        image_transport
        camera_info_manager
        rm_msgs
        cv_bridge
        )
find_package(
        OpenCV REQUIRED
)

generate_dynamic_reconfigure_options(
        cfg/camera.cfg
)

catkin_package(
        LIBRARIES hk_camera
        CATKIN_DEPENDS nodelet roscpp dynamic_reconfigure
        image_transport camera_info_manager
)

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        include/
        ./
)


add_library(hk_camera
        src/hk_camera.cpp
        )


target_link_libraries(hk_camera
        libMvCameraControl.so ${catkin_LIBRARIES} ${Opencv_LIBS})

add_dependencies(hk_camera ${PROJECT_NAME}_gencfg)

#add_executable(hk_camera_node
#        src/hk_camera_node.cpp)
#add_dependencies(hk_camera_node ${PROJECT_NAME}_gencfg)
#target_link_libraries(hk_camera_node
#        libMvCameraControl.so ${catkin_LIBRARIES} ${Opencv_LIBS})
