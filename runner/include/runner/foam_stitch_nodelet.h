#pragma once

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <dynamic_reconfigure/server.h>
#include <runner/FoamStitchConfig.h>  // 由 dynamic_reconfigure 生成
#include <mutex>

namespace runner {

class FoamStitchNodelet : public nodelet::Nodelet {
public:
  FoamStitchNodelet();
  ~FoamStitchNodelet() override;

private:
  // Nodelet 接口
  void onInit() override;

  // 订阅回调
  void imageCb(const sensor_msgs::ImageConstPtr& msg);

  // dynamic_reconfigure 回调
  void reconfigureCallback(FoamStitchConfig& cfg, uint32_t level);

  // 清空当前拼接
  void resetPanorama();

  // 发布当前拼接图
  void publishPanorama(const ros::Time& stamp);

  // ROS 接口
  ros::Subscriber                            sub_;
  ros::Publisher                             pub_;

  // dynamic_reconfigure server
  std::shared_ptr<dynamic_reconfigure::Server<FoamStitchConfig>> dr_srv_;

  std::string output_topic_;

  cv::Mat panorama_;
  cv::Mat last_foam_;
  std::mutex pano_mutex_;
  bool    auto_reset_;
  int     min_shift_, max_shift_, max_width_;

};

}  // namespace runner
