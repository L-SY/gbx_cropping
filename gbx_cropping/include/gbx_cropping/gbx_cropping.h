//
// Created by lsy on 24-12-23.
//

#pragma once

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>

#include <dynamic_reconfigure/server.h>
#include <gbx_cropping/ImageProcessingConfig.h>

#include <mutex>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>

namespace gbx_cropping
{

class GBXCroppingNodelet : public nodelet::Nodelet
{
public:
  GBXCroppingNodelet();
  ~GBXCroppingNodelet();

private:
  virtual void onInit();

  void imageCallback(const sensor_msgs::ImageConstPtr& msg);
  void triggerCB(gbx_cropping::ImageProcessingConfig &config, uint32_t level);

  // Image processing functions
  std::vector<cv::Point2f> sortPoints(const std::vector<cv::Point2f>& pts);
  cv::Mat warpPerspectiveCustom(const cv::Mat& image, const std::vector<cv::Point2f>& pts, int width = 500, int height = 500);
  bool detectAndCrop(const cv::Mat& image, cv::Mat& warped_image, std::vector<cv::Point2f>& centers);

  // Stitching function
  cv::Mat stitchImages(const cv::Mat& image1, const cv::Mat& image2);
  cv::Mat stitchImagesWithOrb(const cv::Mat& image1, const cv::Mat& image2);

  // SSIM calculation
  double computeSSIM(const cv::Mat& img1, const cv::Mat& img2);

  // Publishers
  image_transport::Publisher pub_annotated_;
  image_transport::Publisher pub_stitched_;

  // Subscribers
  image_transport::Subscriber sub_;

  // Dynamic Reconfigure
  dynamic_reconfigure::Server<gbx_cropping::ImageProcessingConfig> server_;
  gbx_cropping::ImageProcessingConfig config_;
  std::mutex param_mutex_;

  // Buffer for images
  std::mutex image_mutex_;
  std::vector<cv::Mat> image_buffer_;
};

} // namespace gbx_cropping
