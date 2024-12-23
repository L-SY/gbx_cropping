//
// Created by lsy on 24-12-23.
//

#include "gbx_cropping/gbx_cropping.h"

#include <pluginlib/class_list_macros.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/String.h>

// Include headers for SSIM
#include <algorithm>
#include <numeric>
#include <cmath>

namespace gbx_cropping
{

GBXCroppingNodelet::GBXCroppingNodelet()
{
}

GBXCroppingNodelet::~GBXCroppingNodelet()
{
}

void GBXCroppingNodelet::onInit()
{
  ros::NodeHandle& nh = getNodeHandle();
  ros::NodeHandle& private_nh = getPrivateNodeHandle();

  image_transport::ImageTransport it(nh);
  sub_ = it.subscribe("/hk_camera/image_raw", 1, &GBXCroppingNodelet::imageCallback, this);

  pub_annotated_ = it.advertise("annotated_image", 1);
  pub_stitched_ = it.advertise("stitched_image", 1);

  // Load the reference image for SSIM
  std::string reference_image_path;
  private_nh.param<std::string>("reference_image_path", reference_image_path, "reference_stitched_image.jpg");
  reference_image_ = cv::imread(reference_image_path, cv::IMREAD_COLOR);
  if (reference_image_.empty())
  {
    NODELET_ERROR("Failed to load reference image from %s", reference_image_path.c_str());
  }
  else
  {
    NODELET_INFO("Loaded reference image from %s", reference_image_path.c_str());
  }
}

std::vector<cv::Point2f> GBXCroppingNodelet::sortPoints(const std::vector<cv::Point2f>& pts)
{
  std::vector<cv::Point2f> sorted;
  if (pts.size() != 4)
  {
    NODELET_ERROR("Cannot sort points, expected 4 points but got %lu", pts.size());
    return sorted;
  }

  // Copy points to a mutable vector
  std::vector<cv::Point2f> points = pts;

  // Initialize variables
  cv::Point2f left_top, right_top, right_bottom, left_bottom;

  // Compute the sum and difference for each point
  std::vector<float> sum_pts;
  std::vector<float> diff_pts;
  for (size_t i = 0; i < points.size(); ++i)
  {
    sum_pts.push_back(points[i].x + points[i].y);
    diff_pts.push_back(points[i].y - points[i].x);
  }

  // Find left_top (smallest sum)
  auto min_sum_it = std::min_element(sum_pts.begin(), sum_pts.end());
  size_t left_top_idx = std::distance(sum_pts.begin(), min_sum_it);
  left_top = points[left_top_idx];

  // Find right_bottom (largest sum)
  auto max_sum_it = std::max_element(sum_pts.begin(), sum_pts.end());
  size_t right_bottom_idx = std::distance(sum_pts.begin(), max_sum_it);
  right_bottom = points[right_bottom_idx];

  // Find right_top (smallest difference)
  auto min_diff_it = std::min_element(diff_pts.begin(), diff_pts.end());
  size_t right_top_idx = std::distance(diff_pts.begin(), min_diff_it);
  right_top = points[right_top_idx];

  // Find left_bottom (largest difference)
  auto max_diff_it = std::max_element(diff_pts.begin(), diff_pts.end());
  size_t left_bottom_idx = std::distance(diff_pts.begin(), max_diff_it);
  left_bottom = points[left_bottom_idx];

  sorted.push_back(left_top);
  sorted.push_back(right_top);
  sorted.push_back(right_bottom);
  sorted.push_back(left_bottom);

  return sorted;
}

cv::Mat GBXCroppingNodelet::warpPerspectiveCustom(const cv::Mat& image, const std::vector<cv::Point2f>& pts, int width, int height)
{
  if (pts.size() != 4)
  {
    NODELET_ERROR("warpPerspectiveCustom requires exactly 4 points");
    return image.clone();
  }

  cv::Mat dst = (cv::Mat_<float>(4,2) <<
                     0, 0,
                 width - 1, 0,
                 width - 1, height - 1,
                 0, height - 1);

  cv::Mat M = cv::getPerspectiveTransform(pts, dst);
  cv::Mat warped;
  cv::warpPerspective(image, warped, M, cv::Size(width, height));

  return warped;
}

bool GBXCroppingNodelet::detectAndCrop(const cv::Mat& image, cv::Mat& warped_image, std::vector<cv::Point2f>& centers)
{
  // Check if image is empty
  if (image.empty())
  {
    NODELET_ERROR("Empty image received for detection");
    return false;
  }

  cv::Mat gray;
  if (image.channels() == 3)
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  else
    gray = image.clone();

  // Apply Gaussian Blur
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2);

  // Adaptive Thresholding (Inverse Binary)
  cv::Mat thresh;
  cv::adaptiveThreshold(
      blurred, thresh,
      255,
      cv::ADAPTIVE_THRESH_GAUSSIAN_C,
      cv::THRESH_BINARY_INV,
      11, 2
  );

  // Morphological Operations (Close and Dilate)
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
  cv::Mat thresh_close;
  cv::morphologyEx(thresh, thresh_close, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 3);
  cv::Mat thresh_dilate;
  cv::morphologyEx(thresh_close, thresh_dilate, cv::MORPH_DILATE, kernel, cv::Point(-1,-1), 3);

  // Find Contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh_dilate, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Draw detected circles on a copy for visualization
  cv::Mat detected_circles_image = image.clone();

  for (size_t i = 0; i < contours.size(); ++i)
  {
    double area = cv::contourArea(contours[i]);
    if (area < 3000 || area > 500000)
      continue;

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contours[i], center, radius);

    double perimeter = cv::arcLength(contours[i], true);
    if (perimeter == 0)
      continue;
    double circularity = 4 * CV_PI * (area / (perimeter * perimeter));
    if (circularity < 0.1)
      continue;

    centers.push_back(center);

    // Draw circle boundary
    cv::circle(detected_circles_image, center, static_cast<int>(radius), cv::Scalar(0, 255, 0), 2);
    // Draw center point
    cv::circle(detected_circles_image, center, 5, cv::Scalar(0, 0, 255), -1);
  }

  // Check number of centers
  if (centers.size() < 4)
  {
    NODELET_WARN("Detected %lu points, expected 4. Proceeding with visualization.", centers.size());
  }

  // Sort points if enough are detected
  std::vector<cv::Point2f> sorted_centers;
  if (centers.size() >= 4)
  {
    sorted_centers = sortPoints(centers);
  }

  // Draw connecting lines
  for (size_t i = 0; i < sorted_centers.size(); ++i)
  {
    cv::line(
        detected_circles_image,
        sorted_centers[i],
        sorted_centers[(i + 1) % sorted_centers.size()],
        cv::Scalar(255, 0, 0), 2
    );
  }

  // Publish annotated image
  cv_bridge::CvImage annotated_msg;
  annotated_msg.header = std_msgs::Header();
  annotated_msg.encoding = sensor_msgs::image_encodings::BGR8;
  annotated_msg.image = detected_circles_image;
  pub_annotated_.publish(annotated_msg.toImageMsg());

  // If enough points are detected, perform cropping
  if (sorted_centers.size() == 4)
  {
    warped_image = warpPerspectiveCustom(image, sorted_centers, 500, 500);
    return true;
  }
  else
  {
    // If less than 4 points are detected, return the original image
    warped_image = image.clone();
    return false;
  }
}

cv::Mat GBXCroppingNodelet::stitchImages(const cv::Mat& image1, const cv::Mat& image2)
{
  if (image1.empty() || image2.empty())
  {
    NODELET_ERROR("One or both images to stitch are empty");
    return cv::Mat();
  }

  cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
  cv::Stitcher::Status status = stitcher->stitch(std::vector<cv::Mat>{image1, image2}, image1); // image1 will contain the stitched image

  if (status != cv::Stitcher::OK)
  {
    NODELET_ERROR("Cannot stitch images, error code = %d", int(status));
    return cv::Mat();
  }

  // Publish stitched image
  cv_bridge::CvImage stitched_msg;
  stitched_msg.header = std_msgs::Header();
  stitched_msg.encoding = sensor_msgs::image_encodings::BGR8;
  stitched_msg.image = image1;
  pub_stitched_.publish(stitched_msg.toImageMsg());

  return image1;
}

double GBXCroppingNodelet::computeSSIM(const cv::Mat& img1, const cv::Mat& img2)
{
  if (img1.empty() || img2.empty())
  {
    NODELET_ERROR("Cannot compute SSIM on empty images");
    return -1.0;
  }

  // Convert images to grayscale
  cv::Mat gray1, gray2;
  if (img1.channels() == 3)
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
  else
    gray1 = img1.clone();

  if (img2.channels() == 3)
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
  else
    gray2 = img2.clone();

  // Ensure images are the same size
  if (gray1.size() != gray2.size())
  {
    cv::resize(gray2, gray2, gray1.size());
  }

  double C1 = 6.5025, C2 = 58.5225;

  cv::Mat I1, I2;
  gray1.convertTo(I1, CV_32F);
  gray2.convertTo(I2, CV_32F);

  cv::Mat mu1, mu2;
  cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
  cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

  cv::Mat mu1_sq = mu1.mul(mu1);
  cv::Mat mu2_sq = mu2.mul(mu2);
  cv::Mat mu1_mu2 = mu1.mul(mu2);

  cv::Mat sigma1_sq, sigma2_sq, sigma12;
  cv::GaussianBlur(I1.mul(I1), sigma1_sq, cv::Size(11, 11), 1.5);
  sigma1_sq -= mu1_sq;

  cv::GaussianBlur(I2.mul(I2), sigma2_sq, cv::Size(11, 11), 1.5);
  sigma2_sq -= mu2_sq;

  cv::GaussianBlur(I1.mul(I2), sigma12, cv::Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;

  // Now calculate SSIM
  cv::Mat t1, t2, t3;
  cv::addWeighted(mu1_mu2, 2, cv::Mat::ones(mu1_mu2.size(), mu1_mu2.type()) * C1, 0, C1, t1); // (2*mu1_mu2 + C1)
  cv::addWeighted(sigma1_sq + sigma2_sq, 2, cv::Mat::ones(sigma1_sq.size(), sigma1_sq.type()) * C2, 0, C2, t2); // (sigma1_sq + sigma2_sq + C2)
  cv::multiply(t1, t2, t3); // (2*mu1_mu2 + C1)*(sigma1_sq + sigma2_sq + C2)

  cv::Mat t4, t5, t6;
  cv::addWeighted(mu1_sq + mu2_sq, 2, cv::Mat::ones(mu1_sq.size(), mu1_sq.type()) * C1, 0, C1, t4); // (mu1_sq + mu2_sq + C1)
  cv::addWeighted(sigma1_sq.mul(sigma2_sq), 2, cv::Mat::ones(sigma1_sq.size(), sigma1_sq.type()) * C2, 0, C2, t5); // (sigma1 * sigma2 + C2)
  cv::multiply(t4, t5, t6); // (mu1_sq + mu2_sq + C1)*(sigma1_sq * sigma2_sq + C2)

  cv::Mat ssim_map;
  cv::divide(t3, t6, ssim_map);

  cv::Scalar mssim = cv::mean(ssim_map);
  return mssim.val[0];
}

void GBXCroppingNodelet::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(mutex_);

  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    // Convert ROS image message to OpenCV image
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    NODELET_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat warped_image;
  std::vector<cv::Point2f> centers;

  bool has_four_points = detectAndCrop(cv_ptr->image, warped_image, centers);

  // If warped_image is valid, add to buffer
  if (!warped_image.empty())
  {
    image_buffer_.push_back(warped_image);
  }

  // If we have two images in buffer, attempt stitching
  if (image_buffer_.size() == 2)
  {
    cv::Mat stitched = stitchImages(image_buffer_[0], image_buffer_[1]);

    if (!stitched.empty() && !reference_image_.empty())
    {
      double similarity = computeSSIM(stitched, reference_image_);
      NODELET_INFO("SSIM similarity: %.4f", similarity);
    }

    // Clear the buffer after stitching
    image_buffer_.clear();
  }
}

PLUGINLIB_EXPORT_CLASS(gbx_cropping::GBXCroppingNodelet, nodelet::Nodelet)

} // namespace gbx_cropping