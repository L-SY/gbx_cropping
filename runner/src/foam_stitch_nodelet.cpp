#include "runner/foam_stitch_nodelet.h"
#include <pluginlib/class_list_macros.h>

namespace runner {

FoamStitchNodelet::FoamStitchNodelet()
    : min_shift_(1), max_shift_(200), max_width_(2000), auto_reset_(false) {}

FoamStitchNodelet::~FoamStitchNodelet() {}

void FoamStitchNodelet::onInit() {
  ros::NodeHandle nh  = getMTNodeHandle();
  ros::NodeHandle pnh = getMTPrivateNodeHandle();

  dr_srv_.reset(new dynamic_reconfigure::Server<FoamStitchConfig>(pnh));
  dr_srv_->setCallback(
      boost::bind(&FoamStitchNodelet::reconfigureCallback, this, _1, _2)
  );

  sub_ = nh.subscribe("/panel_detector/foam_board/image_raw", 1,
                      &FoamStitchNodelet::imageCb, this);
  pub_           = nh.advertise<sensor_msgs::Image>("/panel_detector/foam_board/stitched", 1);
  debug_raw_pub_ = nh.advertise<sensor_msgs::Image> ("/foam_stitch/debug_raw", 1);

  NODELET_INFO("FoamStitchNodelet initialized");
}

void FoamStitchNodelet::reconfigureCallback(FoamStitchConfig& cfg, uint32_t) {
  std::lock_guard<std::mutex> lock(pano_mutex_);
  min_shift_  = cfg.min_shift;
  max_shift_  = cfg.max_shift;
  max_width_  = cfg.max_width;
  auto_reset_ = cfg.auto_reset;
  if (cfg.reset_now) {
    resetPanorama(); last_img_.release();
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
  } catch (cv_bridge::Exception& e) {
    NODELET_ERROR("cv_bridge exception: %s", e.what()); return;
  }
  cv::Mat img = cv_ptr->image;
  if (img.empty()) return;
  debug_raw_pub_.publish(cv_ptr->toImageMsg());

  std::lock_guard<std::mutex> lock(pano_mutex_);

  if (auto_reset_ || panorama_.empty() || last_img_.empty()) {
    panorama_ = img.clone();
    last_img_ = img.clone();
    publishPanorama(msg->header.stamp);
    return;
  }

  if (img.size() != last_img_.size()) {
    NODELET_WARN("Image size mismatch (%dx%d vs %dx%d), resizing to first frame size",
                 img.cols, img.rows,
                 last_img_.cols, last_img_.rows);
    cv::resize(img, img, last_img_.size());
  }

  // 相位相关计算水平位移 dx
  cv::Mat g0, g1;
  cv::cvtColor(last_img_, g0, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img,      g1, cv::COLOR_BGR2GRAY);
  g0.convertTo(g0, CV_32F);
  g1.convertTo(g1, CV_32F);
  cv::Point2d shift = cv::phaseCorrelate(g0, g1);
  int dx = static_cast<int>(std::round(shift.x));

  if (std::abs(dx) < min_shift_ || std::abs(dx) > max_shift_) {
    last_img_ = img.clone();
    return;
  }

  int start_x = std::max(0, dx);
  int new_w   = img.cols - start_x;
  if (new_w > 0) {
    cv::Mat strip = img(cv::Rect(start_x, 0, new_w, img.rows)).clone();
    cv::hconcat(panorama_, strip, panorama_);
    if (panorama_.cols > max_width_) {
      int off = panorama_.cols - max_width_;
      panorama_ = panorama_(cv::Rect(off, 0, max_width_, panorama_.rows)).clone();
    }
    publishPanorama(msg->header.stamp);
  }

  last_img_ = img.clone();
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
