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
  debug_raw_pub_.publish(cv_ptr->toImageMsg());

  std::lock_guard<std::mutex> lock(pano_mutex_);

  // 首帧或重置
  if (auto_reset_ || panorama_.empty() || last_img_.empty()) {
    NODELET_INFO("Initializing panorama with first frame");
    panorama_ = img.clone();
    last_img_ = img.clone();
    publishPanorama(msg->header.stamp);
    return;
  }

  // 尺寸对齐
  if (img.size() != last_img_.size()) {
    NODELET_WARN("Image size mismatch (%dx%d vs %dx%d), resizing to first frame size",
                 img.cols, img.rows,
                 last_img_.cols, last_img_.rows);
    cv::resize(img, img, last_img_.size());
  }

  // 转灰度
  cv::Mat gray_prev, gray_cur;
  cv::cvtColor(last_img_, gray_prev, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img,       gray_cur,  cv::COLOR_BGR2GRAY);

  cv::equalizeHist(gray_prev, gray_prev);
  cv::equalizeHist(gray_cur,  gray_cur);

  debug_gray_prev_pub_.publish(cv_bridge::CvImage(msg->header, "mono8", gray_prev).toImageMsg());
  debug_gray_cur_pub_ .publish(cv_bridge::CvImage(msg->header, "mono8", gray_cur ).toImageMsg());
  // ORB 特征检测与描述子
  static auto orb = cv::ORB::create(1000);
  std::vector<cv::KeyPoint> kp_prev, kp_cur;
  cv::Mat des_prev, des_cur;
  orb->detectAndCompute(gray_prev, cv::noArray(), kp_prev, des_prev);
  orb->detectAndCompute(gray_cur,  cv::noArray(), kp_cur,  des_cur);
  NODELET_INFO("ORB detected %zu keypoints in previous, %zu in current frame",
               kp_prev.size(), kp_cur.size());

  if (des_prev.empty() || des_cur.empty()) {
    NODELET_WARN("No descriptors found (des_prev empty: %d, des_cur empty: %d)",
                 des_prev.empty(), des_cur.empty());
    last_img_ = img.clone();
    return;
  }

  // BFMatcher KNN + Lowe 筛选
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher.knnMatch(des_prev, des_cur, knn_matches, 2);
  size_t total_knn = knn_matches.size();
  NODELET_INFO("KNN found %zu raw matches", total_knn);

  std::vector<cv::DMatch> good_matches;
  for (auto &m : knn_matches) {
    if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance)
      good_matches.push_back(m[0]);
  }
  NODELET_INFO("Filtered to %zu good matches", good_matches.size());

  // 可视化匹配: 前50条
  cv::Mat vis_matches;
  int num_show = std::min<int>(good_matches.size(), 50);
  if (num_show > 0) {
    cv::drawMatches(
        last_img_, kp_prev,
        img,       kp_cur,
        std::vector<cv::DMatch>(good_matches.begin(), good_matches.begin()+num_show),
        vis_matches,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    debug_match_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", vis_matches).toImageMsg());
    NODELET_INFO("Published %d match visualizations", num_show);
  } else {
    NODELET_WARN("No good matches to visualize");
  }

  // 用 RANSAC 根据匹配点估计平移
  double dx = 0;
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
      dx = H.at<double>(0,2);
      NODELET_INFO("Estimated dx = %.2f (inliers: %zu/%zu)",
                   dx,
                   cv::countNonZero(inliers), inliers.rows);
    } else {
      NODELET_ERROR("estimateAffinePartial2D failed: returned empty matrix");
      last_img_ = img.clone();
      return;
    }
  } else {
    NODELET_WARN("Not enough good matches (%zu) for RANSAC, need >=4", good_matches.size());
    last_img_ = img.clone();
    return;
  }

  // 四舍五入并过滤无效偏移
  int idx = static_cast<int>(std::round(dx));
  NODELET_INFO("Rounded dx to %d", idx);
  if (std::abs(idx) < min_shift_ || std::abs(idx) > max_shift_) {
    NODELET_WARN("dx %d out of valid range [%d, %d]", idx, min_shift_, max_shift_);
    last_img_ = img.clone();
    return;
  }

  // 拼接逻辑
  int add_w = std::abs(idx);
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
      int off = panorama_.cols - max_width_;
      panorama_ = panorama_(cv::Rect(off, 0, max_width_, panorama_.rows)).clone();
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
