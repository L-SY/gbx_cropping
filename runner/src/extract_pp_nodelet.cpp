#include "runner/extract_pp_nodelet.h"
#include <algorithm>
#include <pluginlib/class_list_macros.h>

namespace runner {

ExtractPPNodelet::ExtractPPNodelet()
    : maps_initialized_(false), foam_width_ratio_(0.8), foam_height_ratio_(0.5),
      shutdown_(false) {}

ExtractPPNodelet::~ExtractPPNodelet() {
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    shutdown_ = true;
  }
  if (proc_thread_.joinable()) {
    proc_thread_.join();
  }
}

void ExtractPPNodelet::onInit() {
  ros::NodeHandle &nh = getMTNodeHandle();
  ros::NodeHandle &pnh = getMTPrivateNodeHandle();

  // dynamic_reconfigure
  dr_srv_.reset(new dynamic_reconfigure::Server<ExtractPPConfig>(pnh));
  auto cb = boost::bind(&ExtractPPNodelet::reconfigureCallback, this, _1, _2);
  dr_srv_->setCallback(cb);

  // 发布原始 Image
  pub_result_ =
      nh.advertise<sensor_msgs::Image>("/panel_detector/result/image_raw", 1);
  pub_light_panel_ = nh.advertise<sensor_msgs::Image>(
      "/panel_detector/light_panel/image_raw", 1);
  pub_foam_board_ = nh.advertise<sensor_msgs::Image>(
      "/panel_detector/foam_board/image_raw", 1);

  pub_small_ =
      nh.advertise<sensor_msgs::Image>("/panel_detector/debug/small", 1);
  pub_gray_ = nh.advertise<sensor_msgs::Image>("/panel_detector/debug/gray", 1);
  pub_thresh_ =
      nh.advertise<sensor_msgs::Image>("/panel_detector/debug/thresh", 1);
  pub_contours_ =
      nh.advertise<sensor_msgs::Image>("/panel_detector/debug/contours", 1);

  // 订阅
  sub_ =
      nh.subscribe("/hk_camera/image_raw", 1, &ExtractPPNodelet::imageCb, this);

  // 相机标定
  camera_matrix_ = (cv::Mat_<double>(3, 3) << 2343.181585, 0.0, 1221.765641,
                    0.0, 2341.245683, 1040.731733, 0.0, 0.0, 1.0);
  dist_coeffs_ =
      (cv::Mat_<double>(1, 5) << -0.080789, 0.084471, 0.000261, 0.000737, 0.0);

  // 处理线程
  proc_thread_ = std::thread(&ExtractPPNodelet::processingLoop, this);

  NODELET_INFO(
      "ExtractPPNodelet initialized (raw/image + dynamic_reconfigure).");
}

void ExtractPPNodelet::reconfigureCallback(ExtractPPConfig &config, uint32_t) {
  foam_width_ratio_ = config.foam_width_ratio;
  foam_height_ratio_ = config.foam_height_ratio;
  box_smoothing_alpha_ = config.box_smoothing_alpha;
  scale_        = config.scale;
  blur_size_    = config.blur_size;
  use_otsu_     = config.use_otsu;
  thresh_value_ = config.thresh_value;
  area_thresh_  = config.area_thresh;
  ar_tol_       = config.ar_tol;

  NODELET_INFO("Reconf: foam_w=%.2f foam_h=%.2f alpha=%.2f", foam_width_ratio_,
               foam_height_ratio_, box_smoothing_alpha_);
}

void ExtractPPNodelet::imageCb(const sensor_msgs::ImageConstPtr &msg) {
  std::lock_guard<std::mutex> lk(queue_mutex_);
  if (image_queue_.size() < 5) {
    image_queue_.push(msg);
  }
}

void ExtractPPNodelet::processingLoop() {
  ros::Rate rate(100);
  while (ros::ok()) {
    sensor_msgs::ImageConstPtr msg;
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      if (shutdown_)
        break;
      if (!image_queue_.empty()) {
        msg = image_queue_.front();
        image_queue_.pop();
      }
    }
    if (msg) {
      try {
        auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        cv::Mat raw = cv_ptr->image;

        if (!maps_initialized_) {
          initUndistortMaps(raw.size());
        }

        cv::Mat undist;
        cv::remap(raw, undist, map1_, map2_, cv::INTER_LINEAR);

        processImage(undist, msg->header.stamp);
      } catch (const std::exception &e) {
        NODELET_ERROR("Processing exception: %s", e.what());
      }
    }
    rate.sleep();
  }
}

void ExtractPPNodelet::initUndistortMaps(const cv::Size &image_size) {
  cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(),
                              camera_matrix_, image_size, CV_16SC2, map1_,
                              map2_);
  maps_initialized_ = true;
  NODELET_INFO("Undistort maps init: %dx%d", image_size.width,
               image_size.height);
}

void ExtractPPNodelet::processImage(const cv::Mat &img,
                                    const ros::Time &stamp) {
  cv::Mat light_panel;
  std::vector<cv::Point> panel_box;
  cv::Mat M;
  if (!detectLightPanel(img, light_panel, panel_box, M)) {
    cv::Mat res = img.clone();
    cv::putText(res, "Light panel not detected", cv::Point(50, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    publishRaw(res, pub_result_, stamp);
    return;
  }

  std::vector<cv::Point2f> panel_box_f(panel_box.begin(), panel_box.end());
  if (have_prev_panel_box_) {
    for (size_t i = 0; i < panel_box_f.size(); ++i) {
      panel_box_f[i] = box_smoothing_alpha_ * prev_panel_box_[i] +
                       (1.0 - box_smoothing_alpha_) * panel_box_f[i];
    }
  } else {
    have_prev_panel_box_ = true;
  }
  prev_panel_box_ = panel_box_f;

  std::vector<cv::Point> panel_box_smooth;
  for (auto &p : panel_box_f) {
    panel_box_smooth.emplace_back(cvRound(p.x), cvRound(p.y));
  }

  cv::Mat foam;
  std::vector<cv::Point> foam_box;
  extractFoamBoard(light_panel, foam, foam_box);

  std::vector<cv::Point> foam_box_orig =
      transformFoamBoxToOriginal(foam_box, M);

  std::vector<cv::Point2f> foam_box_f(foam_box_orig.begin(),
                                      foam_box_orig.end());
  if (have_prev_foam_box_) {
    for (size_t i = 0; i < foam_box_f.size(); ++i) {
      foam_box_f[i] = box_smoothing_alpha_ * prev_foam_box_[i] +
                      (1.0 - box_smoothing_alpha_) * foam_box_f[i];
    }
  } else {
    have_prev_foam_box_ = true;
  }
  prev_foam_box_ = foam_box_f;
  std::vector<cv::Point> foam_box_smooth;
  for (auto &p : foam_box_f) {
    foam_box_smooth.emplace_back(cvRound(p.x), cvRound(p.y));
  }

  cv::Mat annotated = img.clone();
  cv::drawContours(annotated,
                   std::vector<std::vector<cv::Point>>{panel_box_smooth}, -1,
                   cv::Scalar(0, 255, 0), 2);
  cv::drawContours(annotated,
                   std::vector<std::vector<cv::Point>>{foam_box_smooth}, -1,
                   cv::Scalar(0, 0, 255), 2);

  std::ostringstream oss;
  oss << "FW=" << std::fixed << std::setprecision(2) << foam_width_ratio_
      << " FH=" << std::fixed << std::setprecision(2) << foam_height_ratio_
      << " alpha=" << std::fixed << std::setprecision(2)
      << box_smoothing_alpha_;
  cv::putText(annotated, oss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 2);

  publishRaw(annotated, pub_result_, stamp);
  publishRaw(light_panel, pub_light_panel_, stamp);
  publishRaw(foam, pub_foam_board_, stamp);
}

bool ExtractPPNodelet::detectLightPanel(const cv::Mat &img, cv::Mat &out_warped,
                                        std::vector<cv::Point> &out_box,
                                        cv::Mat &out_M) {
  // 1) 缩放
  cv::Mat small;
  cv::resize(img, small, cv::Size(), scale_, scale_);
  publishRaw(small, pub_small_, ros::Time::now());

  // 2) 灰度
  cv::Mat gray;
  cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
  publishMono(gray, pub_gray_, ros::Time::now());

  // 3) 模糊
  int k = int(blur_size_) | 1; // 确保是奇数
  cv::GaussianBlur(gray, gray, cv::Size(k, k), 0);

  // 4) 二值化
  cv::Mat thresh;
  if (use_otsu_) {
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  } else {
    cv::threshold(gray, thresh, thresh_value_, 255, cv::THRESH_BINARY);
  }
  publishMono(thresh, pub_thresh_, ros::Time::now());

  // 5) 轮廓检测
  std::vector<std::vector<cv::Point>> ctrs;
  cv::findContours(thresh, ctrs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (ctrs.empty())
    return false;

  // 找最大轮廓并过滤面积
  auto max_it =
      std::max_element(ctrs.begin(), ctrs.end(), [](auto &a, auto &b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
  double area = cv::contourArea(*max_it) / (scale_ * scale_);
  if (area < area_thresh_)
    return false;

  // 6) 恢复到原尺寸
  std::vector<cv::Point> contour;
  contour.reserve(max_it->size());
  for (auto &p : *max_it) {
    contour.emplace_back(cvRound(p.x / scale_), cvRound(p.y / scale_));
  }

  // 7) 最小外接矩形
  cv::RotatedRect rect = cv::minAreaRect(contour);

  // 根据平滑后参数判断长宽比
  int W = cvRound(rect.size.width), H = cvRound(rect.size.height);
  double ar = double(std::max(W, H)) / double(std::max(1, std::min(W, H)));
  if (std::abs(ar - 1.0) > ar_tol_)
    return false;

  // 8) 将矩形画到 debug 图上
  cv::Mat dbg = img.clone();
  cv::Point2f ptsf[4];
  rect.points(ptsf);
  for (int i = 0; i < 4; ++i)
    cv::line(dbg, ptsf[i], ptsf[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
  publishRaw(dbg, pub_contours_, ros::Time::now());

  // 9) 角点排序 & 透视变换
  std::vector<cv::Point2f> pts(ptsf, ptsf + 4);
  std::sort(pts.begin(), pts.end(), [](auto &a, auto &b) { return a.y < b.y; });
  std::vector<cv::Point2f> top(pts.begin(), pts.begin() + 2),
      bot(pts.begin() + 2, pts.end());
  std::sort(top.begin(), top.end(), [](auto &a, auto &b) { return a.x < b.x; });
  std::sort(bot.begin(), bot.end(), [](auto &a, auto &b) { return a.x > b.x; });

  std::vector<cv::Point2f> src = {top[0], top[1], bot[0], bot[1]};
  out_box.clear();
  for (auto &p : src)
    out_box.emplace_back(cvRound(p.x), cvRound(p.y));

  int dstSize = std::max(W, H);
  std::vector<cv::Point2f> dst = {{0, 0},
                                  {(float)dstSize - 1, 0},
                                  {(float)dstSize - 1, (float)dstSize - 1},
                                  {0, (float)dstSize - 1}};
  out_M = cv::getPerspectiveTransform(src, dst);
  cv::warpPerspective(img, out_warped, out_M, cv::Size(dstSize, dstSize));
  return true;
}

void ExtractPPNodelet::extractFoamBoard(const cv::Mat &panel, cv::Mat &out_foam,
                                        std::vector<cv::Point> &out_box) {
  int W = panel.cols;
  int H = panel.rows;
  // 按最新运行时比例裁切
  int fw = std::max(1, int(W * foam_width_ratio_));
  int fh = std::max(1, int(H * foam_height_ratio_));
  int left = (W - fw) / 2;
  int top = (H - fh) / 2;
  cv::Rect roi(left, top, fw, fh);

  // 真正裁切
  out_foam = panel(roi).clone();

  // 在 warped panel 上输出 ROI 四点
  out_box = {{roi.x, roi.y},
             {roi.x + fw - 1, roi.y},
             {roi.x + fw - 1, roi.y + fh - 1},
             {roi.x, roi.y + fh - 1}};
}

std::vector<cv::Point> ExtractPPNodelet::transformFoamBoxToOriginal(
    const std::vector<cv::Point> &foam_box, const cv::Mat &M) {
  cv::Mat M_inv = M.inv();
  std::vector<cv::Point> out;
  for (auto &p : foam_box) {
    cv::Mat v = (cv::Mat_<double>(3, 1) << p.x, p.y, 1.0);
    cv::Mat r = M_inv * v;
    double ix = r.at<double>(0, 0) / r.at<double>(2, 0);
    double iy = r.at<double>(1, 0) / r.at<double>(2, 0);
    out.emplace_back(cvRound(ix), cvRound(iy));
  }
  return out;
}

void ExtractPPNodelet::publishRaw(const cv::Mat &img, ros::Publisher &pub,
                                  const ros::Time &t) {
  cv_bridge::CvImage out;
  out.header.stamp = t;
  out.encoding = "bgr8";
  out.image = img;
  pub.publish(out.toImageMsg());
}

void ExtractPPNodelet::publishMono(
    const cv::Mat& img,
    ros::Publisher& pub,
    const ros::Time& t)
{
  // 单通道图像用 mono8
  cv_bridge::CvImage out;
  out.header.stamp = t;
  out.encoding    = "mono8";
  out.image       = img;
  pub.publish(out.toImageMsg());
}

} // namespace runner

PLUGINLIB_EXPORT_CLASS(runner::ExtractPPNodelet, nodelet::Nodelet)
