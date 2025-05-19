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

  // 拼接参数
  int    max_width_;     // 当 panorama 宽度超过该值时，丢掉最左边部分
  bool   auto_reset_;    // 接收到新图像时是否自动 reset
  std::string output_topic_;

  // 拼接状态
  cv::Mat            panorama_;
  std::mutex         pano_mutex_;
  cv::Mat           last_foam_;
  // 动态参数：忽略过小或过大的 shift
  int               min_shift_;
  int               max_shift_;
};

}  // namespace runner
