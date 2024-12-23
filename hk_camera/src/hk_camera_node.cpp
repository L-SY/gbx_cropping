//
// Created by zihan on 2022/6/2.
//
#include <hk_camera.h>
#include <nodelet/loader.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "hk_camera_node");

  nodelet::Loader nodelet;
  const nodelet::M_string& remap(ros::names::getRemappings());
  nodelet::V_string nargv;

  nodelet.load(ros::this_node::getName(), "hk_camera/HKCameraNodelet", remap, nargv);

  ros::spin();
  return 0;
}
