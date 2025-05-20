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
  ros::Publisher pub_;

  std::shared_ptr<dynamic_reconfigure::Server<FoamStitchConfig>> dr_srv_;

  std::string input_topic_, output_topic_;

  cv::Mat panorama_;
  cv::Mat last_roi_;
  std::mutex pano_mutex_;

  // parameters
  bool auto_reset_;
  double scale_, area_thresh_, ar_tol_;
  int blur_size_, min_shift_, max_shift_, max_width_;

  // foam_stitch_nodelet.h
  ros::Publisher debug_raw_pub_;
  ros::Publisher debug_gray_pub_;
  ros::Publisher debug_bin_pub_;
  ros::Publisher debug_roi_pub_;

};

} // namespace runner