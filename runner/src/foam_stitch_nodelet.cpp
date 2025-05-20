#include "runner/foam_stitch_nodelet.h"
#include <pluginlib/class_list_macros.h>

namespace runner {

FoamStitchNodelet::FoamStitchNodelet()
    : auto_reset_(false), scale_(0.5), blur_size_(5),
      area_thresh_(300.0), ar_tol_(0.3),
      min_shift_(1), max_shift_(200), max_width_(2000) {}

FoamStitchNodelet::~FoamStitchNodelet() {}

void FoamStitchNodelet::onInit() {
  ros::NodeHandle nh = getMTNodeHandle();
  ros::NodeHandle pnh = getMTPrivateNodeHandle();

  // load params
  pnh.param("input_topic", input_topic_, std::string("/panel_detector/foam_board/image_raw"));
  pnh.param("output_topic", output_topic_, std::string("/stitched_image"));

  dr_srv_.reset(new dynamic_reconfigure::Server<FoamStitchConfig>(pnh));
  dr_srv_->setCallback(
      boost::bind(&FoamStitchNodelet::reconfigureCallback, this, _1, _2));

  sub_ = nh.subscribe(input_topic_, 1, &FoamStitchNodelet::imageCb, this);
  pub_ = nh.advertise<sensor_msgs::Image>(output_topic_, 1);
  NODELET_INFO("FoamStitchNodelet: subscribed to %s, publishing to %s",
               input_topic_.c_str(), output_topic_.c_str());

  // foam_stitch_nodelet.cpp::onInit()
  sub_ = nh.subscribe(input_topic_, 1, &FoamStitchNodelet::imageCb, this);
  pub_ = nh.advertise<sensor_msgs::Image>(output_topic_, 1);
  debug_raw_pub_  = nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_raw", 1);
  debug_gray_pub_ = nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_gray", 1);
  debug_bin_pub_  = nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_bin", 1);
  debug_roi_pub_  = nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_roi", 1);

}

void FoamStitchNodelet::reconfigureCallback(FoamStitchConfig& cfg, uint32_t) {
  std::lock_guard<std::mutex> lock(pano_mutex_);
  scale_        = cfg.scale;
  blur_size_    = cfg.blur_size;
  area_thresh_  = cfg.area_thresh;
  ar_tol_       = cfg.ar_tol;
  auto_reset_   = cfg.auto_reset;
  min_shift_    = cfg.min_shift;
  max_shift_    = cfg.max_shift;
  max_width_    = cfg.max_width;
  if (cfg.reset_now) {
    resetPanorama();
    last_roi_.release();
    NODELET_INFO("Panorama and history reset");
  }
}

void FoamStitchNodelet::resetPanorama() {
  panorama_.release();
}

void FoamStitchNodelet::imageCb(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
  } catch (...) {
    NODELET_ERROR("cv_bridge exception");
    return;
  }
  cv::Mat img = cv_ptr->image;
  if (img.empty()) return;

  // 1. preprocess: scale, blur, threshold, find foam ROI
  cv::Mat small;
  cv::resize(img, small, cv::Size(), scale_, scale_);
  cv::Mat gray;
  cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(blur_size_|1, blur_size_|1), 0);
  cv::Mat bin;
  cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  double best_area = 0;
  cv::Rect foam_rect;
  for (auto &c : contours) {
    double area = cv::contourArea(c);
    if (area < area_thresh_ * scale_ * scale_) continue;
    cv::Rect r = cv::boundingRect(c);
    double ar = std::abs((double)r.width/r.height - 1.0);
    if (ar > ar_tol_) continue;
    if (area > best_area) { best_area = area; foam_rect = r; }
  }
  if (best_area == 0) return;

  // map back to full size
  cv::Rect roi_full(
      foam_rect.x/scale_, foam_rect.y/scale_,
      foam_rect.width/scale_, foam_rect.height/scale_);
  roi_full &= cv::Rect(0,0, img.cols, img.rows);
  cv::Mat roi = img(roi_full).clone();

  std::lock_guard<std::mutex> lock(pano_mutex_);

  // 2. init or reset
  if (auto_reset_ || panorama_.empty() || last_roi_.empty()) {
    panorama_ = roi.clone();
    last_roi_ = roi.clone();
    publishPanorama(msg->header.stamp);
    return;
  }

  // 3. compute shift via phase correlation
  cv::Mat g0, g1;
  cv::cvtColor(last_roi_, g0, cv::COLOR_BGR2GRAY);
  cv::cvtColor(roi,        g1, cv::COLOR_BGR2GRAY);
  g0.convertTo(g0, CV_32F);
  g1.convertTo(g1, CV_32F);
  cv::Point2d shift = cv::phaseCorrelate(g0, g1);
  int dx = std::round(shift.x);
  if (std::abs(dx) < min_shift_ || std::abs(dx) > max_shift_) {
    last_roi_ = roi.clone();
    return;
  }

  // 4. extract new strip and concat
  int start = std::max(0, dx);
  int w = roi.cols - start;
  if (w > 0) {
    cv::Mat strip = roi(cv::Rect(start,0,w,roi.rows)).clone();
    cv::hconcat(panorama_, strip, panorama_);
    if (panorama_.cols > max_width_) {
      int off = panorama_.cols - max_width_;
      panorama_ = panorama_(cv::Rect(off,0,max_width_, panorama_.rows)).clone();
    }
    publishPanorama(msg->header.stamp);
  }
  last_roi_ = roi.clone();
}

void FoamStitchNodelet::publishPanorama(const ros::Time& stamp) {
  if (panorama_.empty()) return;
  cv_bridge::CvImage out;
  out.header.stamp = stamp;
  out.encoding     = "bgr8";
  out.image        = panorama_;
  pub_.publish(out.toImageMsg());
}

} // namespace runner

PLUGINLIB_EXPORT_CLASS(runner::FoamStitchNodelet, nodelet::Nodelet)
