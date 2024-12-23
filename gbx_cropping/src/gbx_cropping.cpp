#include <gbx_cropping/gbx_cropping.h>
#include <pluginlib/class_list_macros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
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

  // Initialize image transport
  image_transport::ImageTransport it(nh);
  sub_ = it.subscribe("/hk_camera/image_raw", 1, &GBXCroppingNodelet::imageCallback, this);

  // Initialize publishers
  pub_annotated_ = it.advertise("annotated_image", 1);
  pub_stitched_ = it.advertise("stitched_image", 1);

  // Initialize dynamic reconfigure
  dynamic_reconfigure::Server<ImageProcessingConfig>::CallbackType f;
  f = boost::bind(&GBXCroppingNodelet::triggerCB, this, _1, _2);
  server_.setCallback(f);

  // Ensure output directory exists
  {
    std::lock_guard<std::mutex> lock(param_mutex_);
    if (!boost::filesystem::exists(config_.output_directory))
    {
      boost::filesystem::create_directories(config_.output_directory);
      NODELET_INFO("Created output directory: %s", config_.output_directory.c_str());
    }
  }

  NODELET_INFO("Image Stitching Nodelet Initialized");
}

void GBXCroppingNodelet::triggerCB(ImageProcessingConfig &config, uint32_t level)
{
  std::lock_guard<std::mutex> lock(param_mutex_);
  config_ = config;
  NODELET_INFO("Dynamic Reconfigure Request: block_size=%d, C=%.2f, min_area=%.2f, max_area=%.2f, circularity_threshold=%.2f, close_kernel_size=%d, close_iterations=%d, close_operation=%d, dilate_kernel_size=%d, dilate_iterations=%d, dilate_operation=%d, output_directory=%s",
               config_.block_size, config_.C, config_.min_area, config_.max_area,
               config_.circularity_threshold, config_.close_kernel_size, config_.close_iterations,
               config_.close_operation, config_.dilate_kernel_size, config_.dilate_iterations,
               config_.dilate_operation, config_.output_directory.c_str());

  // Ensure output directory exists
  if (!boost::filesystem::exists(config_.output_directory))
  {
    boost::filesystem::create_directories(config_.output_directory);
    NODELET_INFO("Created output directory: %s", config_.output_directory.c_str());
  }
}

std::vector<cv::Point2f> GBXCroppingNodelet::sortPoints(const std::vector<cv::Point2f>& pts)
{
  std::vector<cv::Point2f> sorted;
  if (pts.size() < 1)
  {
    NODELET_ERROR("No points to sort");
    return sorted;
  }

  // Trivial sorting if points are fewer than 4
  if (pts.size() < 4)
  {
    sorted = pts;
    return sorted;
  }

  // Sort the points as per left-top, right-top, right-bottom, left-bottom
  // Calculate the sum and difference
  std::vector<double> sums, diffs;
  for (const auto& pt : pts)
  {
    sums.push_back(pt.x + pt.y);
    diffs.push_back(pt.y - pt.x);
  }

  // Find the left-top point (smallest sum)
  size_t left_top_idx = std::min_element(sums.begin(), sums.end()) - sums.begin();
  cv::Point2f left_top = pts[left_top_idx];

  // Find the right-bottom point (largest sum)
  size_t right_bottom_idx = std::max_element(sums.begin(), sums.end()) - sums.begin();
  cv::Point2f right_bottom = pts[right_bottom_idx];

  // Find the right-top point (smallest difference)
  size_t right_top_idx = std::min_element(diffs.begin(), diffs.end()) - diffs.begin();
  cv::Point2f right_top = pts[right_top_idx];

  // Find the left-bottom point (largest difference)
  size_t left_bottom_idx = std::max_element(diffs.begin(), diffs.end()) - diffs.begin();
  cv::Point2f left_bottom = pts[left_bottom_idx];

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
  cv::warpPerspective(image, warped, M, cv::Size(width, height), cv::INTER_LANCZOS4);

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

  // Convert to grayscale
  cv::Mat gray;
  if (image.channels() == 3)
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  else
    gray = image.clone();

  // Apply Gaussian Blur with dynamic parameters
  cv::Mat blurred;
  {
    std::lock_guard<std::mutex> lock(param_mutex_);
    int block_size = config_.block_size;
    // Ensure block_size is odd and >=3
    if (block_size % 2 == 0)
      block_size += 1;
    block_size = std::max(3, block_size);
    cv::GaussianBlur(gray, blurred, cv::Size(block_size, block_size), 0);
  }

  // Adaptive Thresholding (Inverse Binary) with dynamic parameters
  cv::Mat thresh;
  {
    std::lock_guard<std::mutex> lock(param_mutex_);
    int block_size = config_.block_size;
    if (block_size % 2 == 0)
      block_size += 1;
    block_size = std::max(3, block_size);
    thresh = cv::adaptiveThreshold(
        blurred, thresh,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        block_size, config_.C
    );
  }

  // Morphological Operations with dynamic parameters
  cv::Mat thresh_close, thresh_dilate;
  {
    std::lock_guard<std::mutex> lock(param_mutex_);
    // Closed Morphology
    int close_kernel_size = config_.close_kernel_size;
    close_kernel_size = std::max(1, close_kernel_size);
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(close_kernel_size, close_kernel_size));
    cv::MorphTypes morph_close_type;
    switch(config_.close_operation)
    {
    case 0:
      morph_close_type = cv::MORPH_CLOSE;
      break;
    case 1:
      morph_close_type = cv::MORPH_OPEN;
      break;
    case 2:
      morph_close_type = cv::MORPH_GRADIENT;
      break;
    default:
      NODELET_WARN("Unsupported close_operation: %d. Using MORPH_CLOSE.", config_.close_operation);
      morph_close_type = cv::MORPH_CLOSE;
      break;
    }
    cv::morphologyEx(thresh, thresh_close, morph_close_type, kernel_close, cv::Point(-1,-1), config_.close_iterations);

    // Dilate Morphology
    int dilate_kernel_size = config_.dilate_kernel_size;
    dilate_kernel_size = std::max(1, dilate_kernel_size);
    cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_kernel_size, dilate_kernel_size));
    cv::MorphTypes morph_dilate_type;
    switch(config_.dilate_operation)
    {
    case 0:
      morph_dilate_type = cv::MORPH_DILATE;
      break;
    case 1:
      morph_dilate_type = cv::MORPH_ERODE;
      break;
    case 2:
      morph_dilate_type = cv::MORPH_OPEN;
      break;
    default:
      NODELET_WARN("Unsupported dilate_operation: %d. Using MORPH_DILATE.", config_.dilate_operation);
      morph_dilate_type = cv::MORPH_DILATE;
      break;
    }
    cv::morphologyEx(thresh_close, thresh_dilate, morph_dilate_type, kernel_dilate, cv::Point(-1,-1), config_.dilate_iterations);
  }

  // Find Contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh_dilate, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Draw detected circles on a copy for visualization
  cv::Mat detected_circles_image = image.clone();

  for (size_t i = 0; i < contours.size(); ++i)
  {
    double area = cv::contourArea(contours[i]);
    {
      std::lock_guard<std::mutex> lock(param_mutex_);
      if (area < config_.min_area || area > config_.max_area)
        continue;
    }

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contours[i], center, radius);

    double perimeter = cv::arcLength(contours[i], true);
    if (perimeter == 0)
      continue;
    double circularity = 4 * CV_PI * (area / (perimeter * perimeter));

    {
      std::lock_guard<std::mutex> lock(param_mutex_);
      if (circularity < config_.circularity_threshold)
        continue;
    }

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
  if (centers.size() == 4)
  {
    sorted_centers = sortPoints(centers);
  }
  else if (centers.size() > 4)
  {
    // Sort contours by area in descending order and select top 4
    std::sort(contours.begin(), contours.end(), [&](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) -> bool {
      return cv::contourArea(a) > cv::contourArea(b);
    });

    centers.clear();
    for(int i = 0; i < 4 && i < contours.size(); ++i)
    {
      cv::Point2f center;
      float radius;
      cv::minEnclosingCircle(contours[i], center, radius);
      centers.push_back(center);
    }

    sorted_centers = sortPoints(centers);
  }

  // Draw connecting lines if sorted_centers has at least two points
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
  annotated_msg.header = msg->header; // 保留原始图像的头信息
  annotated_msg.encoding = sensor_msgs::image_encodings::BGR8;
  annotated_msg.image = detected_circles_image;
  pub_annotated_.publish(annotated_msg.toImageMsg());

  // If enough points are detected, perform cropping and save
  if (sorted_centers.size() == 4)
  {
    warped_image = warpPerspectiveCustom(image, sorted_centers, 500, 500);

    // Save cropped image with timestamp
    std::string output_path;
    {
      std::lock_guard<std::mutex> lock(param_mutex_);
      // 使用Boost库形式连接路径
      boost::filesystem::path dir(config_.output_directory);
      boost::filesystem::path filename = "cropped_" + std::to_string(static_cast<long long>(msg->header.stamp.toNSec())) + ".jpg";
      output_path = (dir / filename).string();
    }
    bool success = cv::imwrite(output_path, warped_image);
    if(success)
      NODELET_INFO("Cropped image saved to: %s", output_path.c_str());
    else
      NODELET_ERROR("Failed to save cropped image to: %s", output_path.c_str());

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

  // Using OpenCV's Stitcher and correctly handling cv::Ptr
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
  stitched_msg.header.stamp = ros::Time::now();
  stitched_msg.encoding = sensor_msgs::image_encodings::BGR8;
  stitched_msg.image = image1;
  pub_stitched_.publish(stitched_msg.toImageMsg());

  return image1;
}

cv::Mat GBXCroppingNodelet::stitchImagesWithOrb(const cv::Mat& image1, const cv::Mat& image2)
{
  if (image1.empty() || image2.empty())
  {
    NODELET_ERROR("One or both images to stitch are empty");
    return cv::Mat();
  }

  // Initialize ORB detector with increased features for better matching
  cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);

  // Detect keypoints and compute descriptors
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
  orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

  // Create BFMatcher with Hamming distance
  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors1, descriptors2, matches);

  // Sort matches by distance
  std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) -> bool {
    return a.distance < b.distance;
  });

  // Select top matches (e.g., top 100)
  int numGoodMatches = std::min(100, static_cast<int>(matches.size()));
  matches.resize(numGoodMatches);

  // Extract location of good matches
  std::vector<cv::Point2f> pts1, pts2;
  for(const auto& match : matches)
  {
    pts1.push_back(keypoints1[match.queryIdx].pt);
    pts2.push_back(keypoints2[match.trainIdx].pt);
  }

  // Find homography using RANSAC
  cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, 5.0);
  if (H.empty())
  {
    NODELET_ERROR("Homography computation failed");
    return cv::Mat();
  }

  // Warp image2 to image1's perspective
  cv::Mat warped_image2;
  cv::warpPerspective(image2, warped_image2, H, cv::Size(image1.cols + image2.cols, image1.rows), cv::INTER_LANCZOS4);

  // Place image1 on the stitched canvas
  cv::Mat stitched = warped_image2.clone();
  image1.copyTo(stitched(cv::Rect(0, 0, image1.cols, image1.rows)));

  // Optional: Apply blending to reduce seam artifacts (e.g., multi-band blending)

  // Publish stitched image
  cv_bridge::CvImage stitched_msg;
  stitched_msg.header = std_msgs::Header();
  stitched_msg.header.stamp = ros::Time::now();
  stitched_msg.encoding = sensor_msgs::image_encodings::BGR8;
  stitched_msg.image = stitched;
  pub_stitched_.publish(stitched_msg.toImageMsg());

  return stitched;
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