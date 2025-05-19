#include "runner/foam_stitch_nodelet.h"
#include <pluginlib/class_list_macros.h>

namespace runner {

FoamStitchNodelet::FoamStitchNodelet()
    : max_width_(2000),
      auto_reset_(false)
{}

FoamStitchNodelet::~FoamStitchNodelet()
{}

void FoamStitchNodelet::onInit() {
  ros::NodeHandle& nh  = getMTNodeHandle();
  ros::NodeHandle& pnh = getMTPrivateNodeHandle();

  // dynamic_reconfigure
  dr_srv_.reset(new dynamic_reconfigure::Server<FoamStitchConfig>(pnh));
  dr_srv_->setCallback(
      boost::bind(&FoamStitchNodelet::reconfigureCallback, this, _1, _2)
  );

  // 订阅 foam_board 原始图
  sub_ = nh.subscribe("/panel_detector/foam_board/image_raw",
                      1, &FoamStitchNodelet::imageCb, this);

  // 发布拼接结果
  output_topic_ = "/panel_detector/foam_board/stitched";
  pub_ = nh.advertise<sensor_msgs::Image>(output_topic_, 1);

  NODELET_INFO("FoamStitchNodelet initialized, publishing to %s", output_topic_.c_str());
}

void FoamStitchNodelet::reconfigureCallback(FoamStitchConfig& cfg, uint32_t) {
  std::lock_guard<std::mutex> lk(pano_mutex_);
  max_width_  = cfg.max_width;
  auto_reset_ = cfg.auto_reset;
  min_shift_  = cfg.min_shift;
  max_shift_  = cfg.max_shift;
  if (cfg.reset_now) {
    resetPanorama();
    last_foam_.release();
    NODELET_INFO("Panorama & last_foam reset");
  }
  NODELET_INFO("Reconf: max_w=%d auto_reset=%s min_shift=%d max_shift=%d",
               max_width_, auto_reset_?"T":"F", min_shift_, max_shift_);
}

void FoamStitchNodelet::resetPanorama() {
  panorama_.release();
}

void FoamStitchNodelet::imageCb(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
  } catch (cv_bridge::Exception& e) {
    NODELET_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  const cv::Mat& current = cv_ptr->image;

  std::lock_guard<std::mutex> lk(pano_mutex_);

  // 自动 reset 或首次初始化
  if (auto_reset_ || panorama_.empty() || last_foam_.empty()) {
    panorama_   = current.clone();
    last_foam_  = current.clone();
    publishPanorama(msg->header.stamp);
    return;
  }

  // 1) 转灰度+浮点，用于 phaseCorrelate
  cv::Mat g_prev, g_curr;
  cv::cvtColor(last_foam_, g_prev, cv::COLOR_BGR2GRAY);
  cv::cvtColor(current,     g_curr, cv::COLOR_BGR2GRAY);
  g_prev.convertTo(g_prev, CV_32F);
  g_curr.convertTo(g_curr, CV_32F);

  // 2) 计算 shift (dx, dy)
  cv::Point2d shift = cv::phaseCorrelate(g_prev, g_curr);
  int dx = int(std::round(shift.x));
  // 只关心水平偏移
  if (std::abs(dx) < min_shift_ || std::abs(dx) > max_shift_) {
    // 忽略过小或过大偏移，直接更新 last_foam_ 但不拼接
    last_foam_ = current.clone();
    NODELET_WARN_THROTTLE(5.0,
                          "Computed shift dx=%d ignored (outside [%d, %d])",
                          dx, min_shift_, max_shift_);
    return;
  }

  // 3) 计算新增区域在 current 中的 ROI
  int W = current.cols, H = current.rows;
  int new_start_x = dx > 0 ? dx : 0;  // 向右移动
  int new_w = W - new_start_x;
  if (new_w <= 0) {
    last_foam_ = current.clone();
    return;
  }
  cv::Rect roi(new_start_x, 0, new_w, H);
  cv::Mat new_region = current(roi);

  // 4) 拼接到 panorama_ 右侧
  cv::hconcat(panorama_, new_region, panorama_);

  // 5) 超出宽度裁掉左侧
  if (panorama_.cols > max_width_) {
    int offset = panorama_.cols - max_width_;
    panorama_ = panorama_(cv::Rect(offset, 0,
                                   max_width_, panorama_.rows)).clone();
  }

  // 6) 更新 last_foam_
  last_foam_ = current.clone();

  // 7) 发布
  publishPanorama(msg->header.stamp);

  NODELET_INFO_THROTTLE(2.0,
                        "shift=%d, appended width=%d, pano_width=%d",
                        dx, new_w, panorama_.cols);
}

void FoamStitchNodelet::publishPanorama(const ros::Time& stamp) {
  if (panorama_.empty()) return;
  cv_bridge::CvImage out;
  out.header.stamp = stamp;
  out.encoding     = "bgr8";
  out.image        = panorama_;
  pub_.publish(out.toImageMsg());
}

}  // namespace runner

PLUGINLIB_EXPORT_CLASS(runner::FoamStitchNodelet, nodelet::Nodelet)
