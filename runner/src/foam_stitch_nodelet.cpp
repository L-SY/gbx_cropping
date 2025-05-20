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

  sub_               = nh.subscribe("/panel_detector/foam_board/image_raw", 1,
                      &FoamStitchNodelet::imageCb, this);
  pub_               = nh.advertise<sensor_msgs::Image>("/panel_detector/foam_board/stitched", 1);
  debug_raw_pub_     = nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_raw", 1);
  debug_stitched_pub_= nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_stitched", 1);

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

  // 首帧或重置
  if (auto_reset_ || panorama_.empty() || last_img_.empty()) {
    panorama_ = img.clone();
    last_img_ = img.clone();
    publishPanorama(msg->header.stamp);
    return;
  }

  // 尺寸对齐
  if (img.size() != last_img_.size()) {
    NODELET_WARN("Image size mismatch, resizing to first frame size");
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

  // 过滤偏移范围
  if (std::abs(dx) < min_shift_ || std::abs(dx) > max_shift_) {
    last_img_ = img.clone();
    return;
  }

  // 计算新增区域宽度和位置
  int add_w = std::abs(dx);
  if (add_w > 0 && add_w < img.cols) {
    cv::Mat strip;
    // dx > 0: current image moved right => new region on left
    if (dx > 0) {
      strip = img(cv::Rect(0, 0, add_w, img.rows)).clone();
      cv::hconcat(strip, panorama_, panorama_);
    } else {
      // dx < 0: current image moved left => new region on right
      strip = img(cv::Rect(img.cols - add_w, 0, add_w, img.rows)).clone();
      cv::hconcat(panorama_, strip, panorama_);
    }
    // 裁剪超过最大宽度
    if (panorama_.cols > max_width_) {
      panorama_ = panorama_(cv::Rect(panorama_.cols - max_width_, 0,
                                     max_width_, panorama_.rows)).clone();
    }

    // 可视化：红框=新增，绿框=当前窗口
    cv::Mat vis = panorama_.clone();
    int h = vis.rows;
    if (dx > 0) {
      // 新增在左
      cv::rectangle(vis,
                    cv::Point(0, 0),
                    cv::Point(add_w - 1, h - 1),
                    cv::Scalar(0,0,255), 2);
      cv::rectangle(vis,
                    cv::Point(vis.cols - img.cols, 0),
                    cv::Point(vis.cols - 1, h - 1),
                    cv::Scalar(0,255,0), 2);
    } else {
      // 新增在右
      cv::rectangle(vis,
                    cv::Point(vis.cols - add_w, 0),
                    cv::Point(vis.cols - 1, h - 1),
                    cv::Scalar(0,0,255), 2);
      cv::rectangle(vis,
                    cv::Point(0, 0),
                    cv::Point(img.cols - 1, h - 1),
                    cv::Scalar(0,255,0), 2);
    }
    cv_bridge::CvImage dbg(msg->header, "bgr8", vis);
    debug_stitched_pub_.publish(dbg.toImageMsg());

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
