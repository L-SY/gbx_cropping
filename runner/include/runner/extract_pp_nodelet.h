#pragma once

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>

namespace runner {

class ExtractPPNodelet : public nodelet::Nodelet {
public:
  ExtractPPNodelet();
  ~ExtractPPNodelet() override;

private:
  // Nodelet 接口
  void onInit() override;

  // 回调与处理线程
  void imageCb(const sensor_msgs::ImageConstPtr& msg);
  void processingLoop();

  // 去畸变映射初始化
  void initUndistortMaps(const cv::Size& image_size);

  // 核心图像处理
  void processImage(const cv::Mat& img, const ros::Time& stamp);

  // 各功能模块
  bool detectLightPanel(const cv::Mat& img,
                        cv::Mat& out_warped,
                        std::vector<cv::Point>& out_box,
                        cv::Mat& out_M);

  void extractFoamBoard(const cv::Mat& panel,
                        cv::Mat& out_foam,
                        std::vector<cv::Point>& out_box);

  std::vector<cv::Point> transformFoamBoxToOriginal(
      const std::vector<cv::Point>& foam_box,
      const cv::Mat& M);

  void publishCompressed(const cv::Mat& img,
                         ros::Publisher& pub,
                         const ros::Time& t);

  // ROS 接口
  ros::Subscriber                   sub_;
  ros::Publisher                    pub_result_;
  ros::Publisher                    pub_light_panel_;
  ros::Publisher                    pub_foam_board_;

  // 相机参数与去畸变映射
  cv::Mat                           camera_matrix_;
  cv::Mat                           dist_coeffs_;
  cv::Mat                           map1_;
  cv::Mat                           map2_;
  bool                              maps_initialized_;

  // 多线程与队列
  std::thread                       proc_thread_;
  std::mutex                        queue_mutex_;
  std::queue<sensor_msgs::ImageConstPtr> image_queue_;
  bool                              shutdown_;
};

}  // namespace runner
