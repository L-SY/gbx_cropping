cmake_minimum_required(VERSION 3.0.2)
project(gbx_cropping)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        nodelet
        sensor_msgs
        cv_bridge
        image_transport
        std_msgs
        dynamic_reconfigure
)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

generate_dynamic_reconfigure_options(
        cfg/ImageProcessing.cfg
)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES gbx_cropping
        CATKIN_DEPENDS roscpp nodelet sensor_msgs cv_bridge image_transport std_msgs dynamic_reconfigure
        DEPENDS OpenCV
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

add_library(gbx_cropping
        src/gbx_cropping.cpp
)

target_link_libraries(gbx_cropping
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

add_dependencies(gbx_cropping ${PROJECT_NAME}_gencfg ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

install(TARGETS gbx_cropping
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(FILES gbx_cropping.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)