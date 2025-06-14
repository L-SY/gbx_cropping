#pragma once
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <dynamic_reconfigure/server.h>
#include <runner/FoamStitchConfig.h>
#include <mutex>

namespace runner {

class FoamStitchNodelet : public nodelet::Nodelet {
public:
  FoamStitchNodelet();
  ~FoamStitchNodelet() override;

private:
  void onInit() override;
  void imageCb(const sensor_msgs::ImageConstPtr& msg);
  void reconfigureCallback(FoamStitchConfig& cfg, uint32_t level);
  void resetPanorama();
  void publishPanorama(const ros::Time& stamp);

  ros::Subscriber sub_;
  ros::Publisher   pub_, debug_raw_pub_, debug_stitched_pub_;
  ros::Publisher debug_match_pub_;
  ros::Publisher debug_gray_prev_pub_, debug_gray_cur_pub_;
  std::shared_ptr<dynamic_reconfigure::Server<FoamStitchConfig>> dr_srv_;

  cv::Mat panorama_, last_img_;
  std::mutex pano_mutex_;

  int  min_shift_, max_shift_, max_width_;
  bool auto_reset_, stitch_along_y_;
  double foam_width_ratio_, foam_height_ratio_;
};

} // namespace runner