#pragma once

#include <cv_bridge/cv_bridge.h>
#include <nodelet/nodelet.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

// dynamic_reconfigure
#include <dynamic_reconfigure/server.h>
#include <runner/ExtractPPConfig.h>

#include <mutex>
#include <queue>
#include <thread>

namespace runner {

class ExtractPPNodelet : public nodelet::Nodelet {
public:
  ExtractPPNodelet();
  ~ExtractPPNodelet() override;

private:
  // Nodelet 接口
  void onInit() override;

  // dynamic_reconfigure 回调
  void reconfigureCallback(ExtractPPConfig &config, uint32_t level);

  // 图像回调与处理线程
  void imageCb(const sensor_msgs::ImageConstPtr &msg);
  void processingLoop();

  // 去畸变映射
  void initUndistortMaps(const cv::Size &image_size);

  // 核心处理
  void processImage(const cv::Mat &img, const ros::Time &stamp);

  // 检测与提取模块
  bool detectLightPanel(const cv::Mat &img, cv::Mat &out_warped,
                        std::vector<cv::Point> &out_box, cv::Mat &out_M);

  void extractFoamBoard(const cv::Mat &panel, cv::Mat &out_foam,
                        std::vector<cv::Point> &out_box);

  std::vector<cv::Point>
  transformFoamBoxToOriginal(const std::vector<cv::Point> &foam_box,
                             const cv::Mat &M);

  // 发布原始 Image
  void publishRaw(const cv::Mat &img, ros::Publisher &pub, const ros::Time &t);
  void publishMono(const cv::Mat &img, ros::Publisher &pub, const ros::Time &t);

  // ROS 接口
  ros::Subscriber sub_;
  ros::Publisher pub_result_;
  ros::Publisher pub_light_panel_;
  ros::Publisher pub_foam_board_;

  // dynamic_reconfigure server
  std::shared_ptr<dynamic_reconfigure::Server<ExtractPPConfig>> dr_srv_;

  // 相机参数
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Mat map1_, map2_;
  bool maps_initialized_;

  // 动态参数
  double foam_width_ratio_;
  double foam_height_ratio_;

  double box_smoothing_alpha_;
  bool have_prev_panel_box_{false};
  std::vector<cv::Point2f> prev_panel_box_;
  bool have_prev_foam_box_{false};
  std::vector<cv::Point2f> prev_foam_box_;

  // 多线程
  std::thread proc_thread_;
  std::mutex queue_mutex_;
  std::queue<sensor_msgs::ImageConstPtr> image_queue_;
  bool shutdown_;

  // dynamic parameters
  double scale_, blur_size_, thresh_value_, area_thresh_, ar_tol_;
  bool use_otsu_;

  // debug publishers
  ros::Publisher pub_small_, pub_gray_, pub_thresh_, pub_contours_;
};

} // namespace runner
