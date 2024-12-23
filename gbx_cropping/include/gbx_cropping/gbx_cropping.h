#ifndef GBX_CROPPING_H
#define GBX_CROPPING_H

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <dynamic_reconfigure/server.h>
#include <gbx_cropping/ImageProcessingConfig.h>
#include <opencv2/core/core.hpp>
#include <mutex>

namespace gbx_cropping
{

class GBXCroppingNodelet : public nodelet::Nodelet
{
public:
  virtual void onInit();

private:
  void imageCallback(const sensor_msgs::ImageConstPtr& msg);
  void triggerCB(ImageProcessingConfig &config, uint32_t level);
  bool detectCircles(const cv::Mat& image, std::vector<cv::Point2f>& centers);
  std::vector<cv::Point2f> sortPoints(const std::vector<cv::Point2f>& pts);
  void publishImage(const cv::Mat& image, const image_transport::Publisher& publisher, 
                    const std::string& encoding);

  image_transport::Subscriber sub_;
  image_transport::Publisher pub_gray_;
  image_transport::Publisher pub_blurred_;
  image_transport::Publisher pub_thresh_;
  image_transport::Publisher pub_annotated_;
  
  dynamic_reconfigure::Server<ImageProcessingConfig> server_;
  dynamic_reconfigure::Server<ImageProcessingConfig>::CallbackType f_;
  ImageProcessingConfig config_;
  
  std::mutex param_mutex_;
  sensor_msgs::ImageConstPtr last_msg_;
};

} // namespace gbx_cropping

#endif // GBX_CROPPING_H