#include <gbx_cropping/gbx_cropping.h>
#include <pluginlib/class_list_macros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace gbx_cropping
{

void GBXCroppingNodelet::onInit()
{
  ros::NodeHandle& nh = getNodeHandle();
  ros::NodeHandle& private_nh = getPrivateNodeHandle();

  // Initialize image transport
  image_transport::ImageTransport it(nh);

  // 订阅原始图像
  sub_ = it.subscribe("/hk_camera/image_raw", 1, &GBXCroppingNodelet::imageCallback, this);

  // 初始化所有图像发布器
  pub_gray_ = it.advertise("/gbx_cropping/gray_image", 1);
  pub_blurred_ = it.advertise("/gbx_cropping/blurred_image", 1);
  pub_thresh_ = it.advertise("/gbx_cropping/threshold_image", 1);
  pub_annotated_ = it.advertise("/gbx_cropping/annotated_image", 1);

  // Initialize dynamic reconfigure
  f_ = boost::bind(&GBXCroppingNodelet::triggerCB, this, _1, _2);
  server_.setCallback(f_);

  NODELET_INFO("Circle Detection Nodelet Initialized");
}

void GBXCroppingNodelet::triggerCB(ImageProcessingConfig &config, uint32_t level)
{
  std::lock_guard<std::mutex> lock(param_mutex_);
  config_ = config;
  NODELET_INFO("Reconfigure Request: block_size=%d, C=%.2f, blur_size=%d, min_area=%.2f, max_area=%.2f, circularity_threshold=%.2f",
               config_.block_size,
               config_.C,
               config_.blur_size,
               config_.min_area,
               config_.max_area,
               config_.circularity_threshold);
}

void GBXCroppingNodelet::publishImage(const cv::Mat& image, const image_transport::Publisher& publisher,
                                      const std::string& encoding)
{
  cv_bridge::CvImage img_msg;
  img_msg.header = last_msg_->header;
  img_msg.encoding = encoding;
  img_msg.image = image;
  publisher.publish(img_msg.toImageMsg());
}

bool GBXCroppingNodelet::detectCircles(const cv::Mat& image, std::vector<cv::Point2f>& centers)
{
  ImageProcessingConfig config;
  {
    std::lock_guard<std::mutex> lock(param_mutex_);
    config = config_;
  }

  // Convert to grayscale
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  publishImage(gray, pub_gray_, sensor_msgs::image_encodings::MONO8);

  // Apply Gaussian Blur
  cv::Mat blurred;
  int blur_size = config.blur_size;
  if (blur_size % 2 == 0) blur_size++; // 确保是奇数
  blur_size = std::max(3, blur_size); // 确保至少为3
  cv::GaussianBlur(gray, blurred, cv::Size(blur_size, blur_size), 0);
  publishImage(blurred, pub_blurred_, sensor_msgs::image_encodings::MONO8);

  // Adaptive Threshold
  cv::Mat thresh;
  int block_size = config.block_size;
  if (block_size % 2 == 0) block_size++; // 确保是奇数
  block_size = std::max(3, block_size); // 确保至少为3

  NODELET_DEBUG("Using block_size: %d", block_size); // 添加调试信息

  try {
    cv::adaptiveThreshold(blurred, thresh, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV,
                          block_size, config.C);
  } catch (cv::Exception& e) {
    NODELET_ERROR("OpenCV error in adaptiveThreshold: %s", e.what());
    return false;
  }

  publishImage(thresh, pub_thresh_, sensor_msgs::image_encodings::MONO8);

  // Find Contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Process contours to find circles
  centers.clear();
  cv::Mat detected_circles_image = image.clone();

  for (const auto& contour : contours)
  {
    double area = cv::contourArea(contour);
    if (area < config.min_area || area > config.max_area)
      continue;

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contour, center, radius);

    double perimeter = cv::arcLength(contour, true);
    if (perimeter == 0)
      continue;

    double circularity = 4 * CV_PI * (area / (perimeter * perimeter));
    if (circularity < config.circularity_threshold)
      continue;

    centers.push_back(center);

    // Draw detected circle and its properties
    cv::circle(detected_circles_image, center, static_cast<int>(radius), cv::Scalar(0, 255, 0), 2);
    cv::circle(detected_circles_image, center, 5, cv::Scalar(0, 0, 255), -1);

    // 在圆周围显示参数信息
    std::string info = cv::format("A=%.0f, C=%.2f", area, circularity);
    cv::putText(detected_circles_image, info,
                cv::Point(center.x - radius, center.y - radius - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }

  // 显示检测到的圆的数量
  cv::putText(detected_circles_image,
              cv::format("Detected: %zu circles", centers.size()),
              cv::Point(20, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

  // If exactly 4 circles found, sort and connect them
  if (centers.size() == 4)
  {
    std::vector<cv::Point2f> sorted_centers = sortPoints(centers);
    for (size_t i = 0; i < sorted_centers.size(); ++i)
    {
      cv::line(detected_circles_image,
               sorted_centers[i],
               sorted_centers[(i + 1) % sorted_centers.size()],
               cv::Scalar(255, 0, 0), 2);

      // 标注顶点顺序
      cv::putText(detected_circles_image,
                  std::to_string(i + 1),
                  cv::Point(sorted_centers[i].x + 10, sorted_centers[i].y + 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }
    centers = sorted_centers;
  }

  publishImage(detected_circles_image, pub_annotated_, sensor_msgs::image_encodings::BGR8);

  return centers.size() == 4;
}

void GBXCroppingNodelet::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  last_msg_ = msg;
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    NODELET_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  std::vector<cv::Point2f> centers;
  bool detected = detectCircles(cv_ptr->image, centers);
  if (detected)
  {
    NODELET_INFO("Successfully detected 4 circles");
  }
}

std::vector<cv::Point2f> GBXCroppingNodelet::sortPoints(const std::vector<cv::Point2f>& pts)
{
  std::vector<cv::Point2f> sorted;
  if (pts.size() != 4)
  {
    NODELET_WARN("sortPoints requires exactly 4 points, got %zu", pts.size());
    return pts;
  }

  // 计算质心
  cv::Point2f center(0, 0);
  for (const auto& pt : pts)
  {
    center += pt;
  }
  center.x /= pts.size();
  center.y /= pts.size();

  // 根据点到质心的角度排序
  std::vector<std::pair<float, cv::Point2f>> angles;
  for (const auto& pt : pts)
  {
    float angle = std::atan2(pt.y - center.y, pt.x - center.x);
    angles.push_back(std::make_pair(angle, pt));
  }

  // 按角度排序
  std::sort(angles.begin(), angles.end(),
            [](const std::pair<float, cv::Point2f>& a,
               const std::pair<float, cv::Point2f>& b) {
              return a.first < b.first;
            });

  // 提取排序后的点
  for (const auto& p : angles)
  {
    sorted.push_back(p.second);
  }

  return sorted;
}

PLUGINLIB_EXPORT_CLASS(gbx_cropping::GBXCroppingNodelet, nodelet::Nodelet)

} // namespace gbx_cropping