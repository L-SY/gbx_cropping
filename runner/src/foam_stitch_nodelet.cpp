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
  // 1) 转 CvImage 并获得 foam patch
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
  } catch (cv_bridge::Exception& e) {
    NODELET_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  // 这里的 current 已经是裁切后的 foam_board patch
  cv::Mat current = cv_ptr->image;

  std::lock_guard<std::mutex> lk(pano_mutex_);

  // 2) 首帧或自动重置：直接初始化 panorama_ 和 last_foam_
  if (auto_reset_ || panorama_.empty() || last_foam_.empty()) {
    panorama_   = current.clone();
    last_foam_ = current.clone();
    publishPanorama(msg->header.stamp);
    return;
  }

  // 3) 如果尺寸仅小范围不同，则 resize；大差异则重置 last_foam_
  cv::Size prev_sz = last_foam_.size();
  cv::Size cur_sz  = current.size();
  int dw = std::abs(prev_sz.width  - cur_sz.width);
  int dh = std::abs(prev_sz.height - cur_sz.height);
  const int ALLOW_PIXELS = 5;
  if ((dw > 0 || dh > 0) && dw <= ALLOW_PIXELS && dh <= ALLOW_PIXELS) {
    cv::resize(current, current, prev_sz);
    NODELET_DEBUG("Resized foam patch from %dx%d to %dx%d",
                  cur_sz.width, cur_sz.height,
                  prev_sz.width, prev_sz.height);
  }
  else if (dw > ALLOW_PIXELS || dh > ALLOW_PIXELS) {
    NODELET_WARN("Foam patch size mismatch too large (%dx%d vs %dx%d), resetting last_patch",
                 prev_sz.width, prev_sz.height,
                 cur_sz.width,  cur_sz.height);
    last_foam_ = current.clone();
    return;
  }

  // 4) 转灰度并浮点化，用于 Phase Correlate
  cv::Mat g0, g1;
  cv::cvtColor(last_foam_, g0, cv::COLOR_BGR2GRAY);
  cv::cvtColor(current,     g1, cv::COLOR_BGR2GRAY);
  g0.convertTo(g0, CV_32F);
  g1.convertTo(g1, CV_32F);

  // 5) 计算水平偏移 dx
  cv::Point2d shift = cv::phaseCorrelate(g0, g1);
  int dx = int(std::round(shift.x));

  // 6) 过滤过小或过大位移
  if (std::abs(dx) < min_shift_ || std::abs(dx) > max_shift_) {
    last_foam_ = current.clone();
    NODELET_WARN_THROTTLE(5.0,
                          "Computed shift dx=%d ignored (outside [%d, %d])",
                          dx, min_shift_, max_shift_);
    return;
  }

  // 7) 计算新增区域 ROI 并截取
  int W = current.cols;
  int H = current.rows;
  int start_x = std::max(0, dx);
  int new_w   = W - start_x;
  if (new_w <= 0) {
    last_foam_ = current.clone();
    return;
  }
  cv::Mat new_strip = current(cv::Rect(start_x, 0, new_w, H)).clone();

  // 8) 拼接到 panorama_ 右侧
  cv::hconcat(panorama_, new_strip, panorama_);

  // 9) 如果超出最大宽度，裁掉最左侧
  if (panorama_.cols > max_width_) {
    int off = panorama_.cols - max_width_;
    panorama_ = panorama_(cv::Rect(off, 0, max_width_, panorama_.rows)).clone();
  }

  // 10) 更新 last_foam_ 并发布全景
  last_foam_ = current.clone();
  publishPanorama(msg->header.stamp);

  NODELET_INFO_THROTTLE(2.0,
                        "dx=%d, appended_width=%d, pano_width=%d",
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
