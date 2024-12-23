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

#include <mutex>
#include <vector>

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

  // Image processing functions
  std::vector<cv::Point2f> sortPoints(const std::vector<cv::Point2f>& pts);
  cv::Mat warpPerspectiveCustom(const cv::Mat& image, const std::vector<cv::Point2f>& pts, int width = 500, int height = 500);
  bool detectAndCrop(const cv::Mat& image, cv::Mat& warped_image, std::vector<cv::Point2f>& centers);

  // Stitching function
  cv::Mat stitchImages(const cv::Mat& image1, const cv::Mat& image2);

  // SSIM calculation
  double computeSSIM(const cv::Mat& img1, const cv::Mat& img2);

  // Publishers
  image_transport::Publisher pub_annotated_;
  image_transport::Publisher pub_stitched_;

  // Subscribers
  image_transport::Subscriber sub_;

  // Buffer for images
  std::mutex mutex_;
  std::vector<cv::Mat> image_buffer_;

  // Reference image for SSIM
  cv::Mat reference_image_;
};

} // namespace gbx_cropping
