#include "runner/foam_stitch_nodelet.h"
#include <pluginlib/class_list_macros.h>

namespace runner {

FoamStitchNodelet::FoamStitchNodelet()
    : min_shift_(1), max_shift_(200), max_width_(2000), auto_reset_(false),foam_width_ratio_(0.7), foam_height_ratio_(0.8) {}

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
  debug_match_pub_ = nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_matches", 1);
  debug_gray_prev_pub_ = nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_gray_prev", 1);
  debug_gray_cur_pub_  = nh.advertise<sensor_msgs::Image>("/foam_stitch/debug_gray_cur",  1);

  NODELET_INFO("FoamStitchNodelet initialized");
}

void FoamStitchNodelet::reconfigureCallback(FoamStitchConfig& cfg, uint32_t) {
  std::lock_guard<std::mutex> lock(pano_mutex_);
  min_shift_  = cfg.min_shift;
  max_shift_  = cfg.max_shift;
  max_width_  = cfg.max_width;
  auto_reset_ = cfg.auto_reset;
  foam_width_ratio_ = cfg.foam_width_ratio;
  foam_height_ratio_= cfg.foam_height_ratio;
  stitch_along_y_   = cfg.stitch_along_y;
  if (cfg.reset_now) {
    resetPanorama(); last_img_.release();
    NODELET_INFO("Panorama and history reset");
  }
}

void FoamStitchNodelet::resetPanorama() {
  panorama_.release();
}

// Updated FoamStitchNodelet::imageCb with additional debug logging
void FoamStitchNodelet::imageCb(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
  } catch (cv_bridge::Exception& e) {
    NODELET_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat img = cv_ptr->image;
  if (img.empty()) {
    NODELET_WARN("Received empty image");
    return;
  }

  // 1. 计算 ROI 并发布带框调试图
  int W = img.cols, H = img.rows;
  int cw = std::max(1, int(W * foam_width_ratio_));
  int ch = std::max(1, int(H * foam_height_ratio_));
  int x0 = (W - cw) / 2;
  int y0 = (H - ch) / 2;
  cv::Rect foam_roi(x0, y0, cw, ch);

  cv::Mat annotated = img.clone();
  cv::rectangle(annotated, foam_roi, cv::Scalar(255,0,0), 2);
  debug_raw_pub_.publish(
      cv_bridge::CvImage(msg->header, "bgr8", annotated).toImageMsg());

  cv::Mat to_stitch = img(foam_roi).clone();

  std::lock_guard<std::mutex> lock(pano_mutex_);

  // 2. 首帧或重置时初始化全景
  if (auto_reset_ || panorama_.empty() || last_img_.empty()) {
    NODELET_INFO("Initializing panorama with first frame");
    panorama_ = to_stitch.clone();
    last_img_ = to_stitch.clone();
    publishPanorama(msg->header.stamp);
    return;
  }

  // 3. 如果尺寸不一致则统一到上一帧大小
  if (to_stitch.size() != last_img_.size()) {
    NODELET_WARN("Size mismatch (%dx%d vs %dx%d), resizing",
                 to_stitch.cols, to_stitch.rows,
                 last_img_.cols, last_img_.rows);
    cv::resize(to_stitch, to_stitch, last_img_.size());
  }

  // 4. 灰度转换 & 直方图均衡
  cv::Mat gray_prev, gray_cur;
  cv::cvtColor(last_img_, gray_prev, cv::COLOR_BGR2GRAY);
  cv::cvtColor(to_stitch,    gray_cur,  cv::COLOR_BGR2GRAY);
  cv::equalizeHist(gray_prev, gray_prev);
  cv::equalizeHist(gray_cur,  gray_cur);

  debug_gray_prev_pub_.publish(
      cv_bridge::CvImage(msg->header, "mono8", gray_prev).toImageMsg());
  debug_gray_cur_pub_ .publish(
      cv_bridge::CvImage(msg->header, "mono8", gray_cur ).toImageMsg());

  // 5. ORB 特征检测 & 描述
  static auto orb = cv::ORB::create(1000);
  std::vector<cv::KeyPoint> kp_prev, kp_cur;
  cv::Mat des_prev, des_cur;
  orb->detectAndCompute(gray_prev, cv::noArray(), kp_prev, des_prev);
  orb->detectAndCompute(gray_cur,  cv::noArray(), kp_cur,  des_cur);
  NODELET_INFO("ORB detected %zu keypoints in prev, %zu in cur",
               kp_prev.size(), kp_cur.size());

  if (des_prev.empty() || des_cur.empty()) {
    NODELET_WARN("No descriptors (prev: %d, cur: %d)",
                 des_prev.empty(), des_cur.empty());
    last_img_ = to_stitch.clone();
    return;
  }

  // 6. BFMatcher KNN + Lowe 筛选
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher.knnMatch(des_prev, des_cur, knn_matches, 2);

  std::vector<cv::DMatch> good_matches;
  for (auto &m : knn_matches) {
    if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance)
      good_matches.push_back(m[0]);
  }
  NODELET_INFO("Filtered to %zu good matches", good_matches.size());

  // 7. 匹配可视化
  cv::Mat vis_matches;
  int num_show = std::min<int>(good_matches.size(), 50);
  if (num_show > 0) {
    cv::drawMatches(
        last_img_, kp_prev,
        to_stitch, kp_cur,
        std::vector<cv::DMatch>(good_matches.begin(), good_matches.begin()+num_show),
        vis_matches, {}, {}, {}, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    debug_match_pub_.publish(
        cv_bridge::CvImage(msg->header, "bgr8", vis_matches).toImageMsg());
    NODELET_INFO("Published %d match visualizations", num_show);
  } else {
    NODELET_WARN("No good matches to visualize");
  }

  // 8. RANSAC 估计仿射矩阵并提取偏移量
  double offset = 0;
  if (good_matches.size() >= 4) {
    std::vector<cv::Point2f> pts_prev, pts_cur;
    for (auto &m : good_matches) {
      pts_prev.push_back(kp_prev[m.queryIdx].pt);
      pts_cur .push_back(kp_cur [m.trainIdx].pt);
    }
    cv::Mat inliers;
    cv::Mat H = cv::estimateAffinePartial2D(
        pts_prev, pts_cur, inliers,
        cv::RANSAC, 3.0);
    if (!H.empty()) {
      offset = stitch_along_y_
                   ? H.at<double>(1,2)   // 纵向拼接取 dy
                   : H.at<double>(0,2);  // 横向拼接取 dx
      NODELET_INFO("Estimated offset = %.2f (inliers %d/%d)",
                   offset,
                   cv::countNonZero(inliers), inliers.rows);
    } else {
      NODELET_ERROR("estimateAffinePartial2D failed");
      last_img_ = to_stitch.clone();
      return;
    }
  } else {
    NODELET_WARN("Not enough matches (%zu) for RANSAC", good_matches.size());
    last_img_ = to_stitch.clone();
    return;
  }

  // 9. 四舍五入 & 范围过滤
  int shift = static_cast<int>(std::round(offset));
  if (std::abs(shift) < min_shift_ || std::abs(shift) > max_shift_) {
    NODELET_WARN("shift %d out of valid range [%d, %d]", shift, min_shift_, max_shift_);
    last_img_ = to_stitch.clone();
    return;
  }

  // 10. 拼接前 Y 模式校验：保证 cols/type 一致
  if (stitch_along_y_) {
    if (panorama_.cols != to_stitch.cols) {
      NODELET_WARN("Y-mode: panorama.cols(%d) != to_stitch.cols(%d), resetting",
                   panorama_.cols, to_stitch.cols);
      panorama_ = to_stitch.clone();
      last_img_ = to_stitch.clone();
      publishPanorama(msg->header.stamp);
      return;
    }
    if (panorama_.type() != to_stitch.type()) {
      panorama_.convertTo(panorama_, to_stitch.type());
    }
  }

  // 11. 执行拼接（hconcat 或 vconcat）并裁剪 & 可视化
  int add_len = std::abs(shift);
  if (add_len > 0) {
    cv::Mat strip, vis = panorama_.clone();

    if (!stitch_along_y_) {
      // —— 横向拼接 hconcat ——
      if (shift > 0) {
        strip = to_stitch(cv::Rect(0, 0, add_len, to_stitch.rows)).clone();
        cv::hconcat(strip, panorama_, panorama_);
      } else {
        strip = to_stitch(cv::Rect(to_stitch.cols - add_len, 0, add_len, to_stitch.rows)).clone();
        cv::hconcat(panorama_, strip, panorama_);
      }
      // 裁剪超出宽度
      if (panorama_.cols > max_width_) {
        int off = panorama_.cols - max_width_;
        panorama_ = panorama_(cv::Rect(off, 0, max_width_, panorama_.rows)).clone();
      }
      // 红/绿框可视化
      if (shift > 0) {
        cv::rectangle(vis, {0,0}, {add_len-1,vis.rows-1},                  {0,0,255},2);
        cv::rectangle(vis, {vis.cols - to_stitch.cols,0}, {vis.cols-1,vis.rows-1}, {0,255,0},2);
      } else {
        cv::rectangle(vis, {vis.cols - add_len,0}, {vis.cols-1,vis.rows-1},       {0,0,255},2);
        cv::rectangle(vis, {0,0}, {to_stitch.cols-1,vis.rows-1},                 {0,255,0},2);
      }
    } else {
      // —— 纵向拼接 vconcat ——
      if (shift > 0) {
        strip = to_stitch(cv::Rect(0, 0, to_stitch.cols, add_len)).clone();
        cv::vconcat(strip, panorama_, panorama_);
      } else {
        strip = to_stitch(cv::Rect(0, to_stitch.rows - add_len, to_stitch.cols, add_len)).clone();
        cv::vconcat(panorama_, strip, panorama_);
      }
      // （如需裁剪高度，可按横向同理添加）
      // 红/绿框可视化
      if (shift > 0) {
        cv::rectangle(vis, {0,0}, {vis.cols-1,add_len-1},                     {0,0,255},2);
        cv::rectangle(vis, {0,vis.rows - to_stitch.rows}, {vis.cols-1,vis.rows-1}, {0,255,0},2);
      } else {
        cv::rectangle(vis, {0,vis.rows - add_len}, {vis.cols-1,vis.rows-1},      {0,0,255},2);
        cv::rectangle(vis, {0,0}, {vis.cols-1,to_stitch.rows-1},                {0,255,0},2);
      }
    }

    debug_stitched_pub_.publish(
        cv_bridge::CvImage(msg->header, "bgr8", vis).toImageMsg());
    publishPanorama(msg->header.stamp);
  }

  // 12. 更新上一帧
  last_img_ = to_stitch.clone();
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
