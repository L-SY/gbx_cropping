#include "runner/extract_pp_nodelet.h"
#include <pluginlib/class_list_macros.h>
#include <algorithm>

namespace runner {

ExtractPPNodelet::ExtractPPNodelet()
    : maps_initialized_(false),
      shutdown_(false)
{}

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
  ros::NodeHandle& nh = getMTNodeHandle();

  // 原始 Image 发布
  pub_result_      = nh.advertise<sensor_msgs::Image>(
      "/panel_detector/result/image_raw", 1);
  pub_light_panel_ = nh.advertise<sensor_msgs::Image>(
      "/panel_detector/light_panel/image_raw", 1);
  pub_foam_board_  = nh.advertise<sensor_msgs::Image>(
      "/panel_detector/foam_board/image_raw", 1);

  // 订阅原始图像
  sub_ = nh.subscribe("/hk_camera/image_raw", 1,
                      &ExtractPPNodelet::imageCb, this);

  // 相机标定参数
  camera_matrix_ = (cv::Mat_<double>(3,3) <<
                        2343.181585, 0.0,         1221.765641,
                    0.0,         2341.245683, 1040.731733,
                    0.0,         0.0,         1.0);
  dist_coeffs_ = (cv::Mat_<double>(1,5) <<
                      -0.080789, 0.084471, 0.000261, 0.000737, 0.0);

  // 启动处理线程
  proc_thread_ = std::thread(&ExtractPPNodelet::processingLoop, this);

  NODELET_INFO("ExtractPPNodelet (raw image) initialized.");
}

void ExtractPPNodelet::imageCb(const sensor_msgs::ImageConstPtr& msg) {
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
      if (shutdown_) break;
      if (!image_queue_.empty()) {
        msg = image_queue_.front();
        image_queue_.pop();
      }
    }
    if (msg) {
      try {
        // 零拷贝获取 OpenCV 图
        auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        cv::Mat raw = cv_ptr->image;

        // 第一次初始化畸变映射
        if (!maps_initialized_) {
          initUndistortMaps(raw.size());
        }

        // 去畸变
        cv::Mat undist;
        cv::remap(raw, undist, map1_, map2_, cv::INTER_LINEAR);

        // 核心处理 & 发布
        processImage(undist, msg->header.stamp);
      }
      catch (const std::exception& e) {
        NODELET_ERROR("Processing exception: %s", e.what());
      }
    }
    rate.sleep();
  }
}

void ExtractPPNodelet::initUndistortMaps(const cv::Size& image_size) {
  cv::initUndistortRectifyMap(
      camera_matrix_, dist_coeffs_, cv::Mat(),
      camera_matrix_, image_size,
      CV_16SC2, map1_, map2_);
  maps_initialized_ = true;
  NODELET_INFO("Undistort maps init: %dx%d",
               image_size.width, image_size.height);
}

void ExtractPPNodelet::processImage(const cv::Mat& img, const ros::Time& stamp) {
  // 1) Light Panel 检测
  cv::Mat light_panel;
  std::vector<cv::Point> panel_box;
  cv::Mat M;
  if (!detectLightPanel(img, light_panel, panel_box, M)) {
    cv::Mat res = img.clone();
    cv::putText(res, "Light panel not detected",
                cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0,0,255), 2);
    publishRaw(res, pub_result_, stamp);
    return;
  }

  // 2) Foam Board 提取
  cv::Mat foam;
  std::vector<cv::Point> foam_box;
  extractFoamBoard(light_panel, foam, foam_box);

  // 3) 反透视变换
  auto foam_box_orig = transformFoamBoxToOriginal(foam_box, M);

  // 4) 绘制并发布所有原始图
  cv::Mat annotated = img.clone();
  cv::drawContours(annotated, std::vector<std::vector<cv::Point>>{panel_box},
                   -1, cv::Scalar(0,255,0), 2);
  cv::drawContours(annotated, std::vector<std::vector<cv::Point>>{foam_box_orig},
                   -1, cv::Scalar(0,0,255), 2);
  cv::putText(annotated, "Light Panel",
              panel_box[0] - cv::Point(0,10),
              cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0,255,0), 2);
  cv::putText(annotated, "Foam Board",
              foam_box_orig[0] - cv::Point(0,10),
              cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0,0,255), 2);

  publishRaw(annotated,   pub_result_,      stamp);
  publishRaw(light_panel, pub_light_panel_, stamp);
  publishRaw(foam,        pub_foam_board_,  stamp);
}

bool ExtractPPNodelet::detectLightPanel(
    const cv::Mat& img,
    cv::Mat& out_warped,
    std::vector<cv::Point>& out_box,
    cv::Mat& out_M)
{
  double scale = 0.5;
  cv::Mat small, gray, blur, thresh;
  cv::resize(img, small, cv::Size(), scale, scale);
  cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blur, cv::Size(5,5), 0);
  cv::threshold(blur, thresh, 0, 255,
                cv::THRESH_BINARY | cv::THRESH_OTSU);

  std::vector<std::vector<cv::Point>> ctrs;
  cv::findContours(thresh, ctrs,
                   cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  if (ctrs.empty()) return false;

  auto max_it = std::max_element(
      ctrs.begin(), ctrs.end(),
      [](auto&a, auto&b){
        return cv::contourArea(a) < cv::contourArea(b);
      });
  if (cv::contourArea(*max_it) < 300) return false;

  std::vector<cv::Point> contour;
  for (auto& p : *max_it) {
    contour.emplace_back(cvRound(p.x/scale),
                         cvRound(p.y/scale));
  }

  cv::RotatedRect rect = cv::minAreaRect(contour);
  cv::Point2f ptsf[4]; rect.points(ptsf);
  std::vector<cv::Point2f> pts(ptsf, ptsf+4);

  int W = cvRound(rect.size.width),
      H = cvRound(rect.size.height);
  double ar = double(std::max(W,H)) /
              double(std::max(1,std::min(W,H)));
  if (std::abs(ar - 1.0) > 0.3) return false;

  std::sort(pts.begin(), pts.end(),
            [](auto&a, auto&b){ return a.y < b.y; });
  std::vector<cv::Point2f> top(pts.begin(), pts.begin()+2),
      bot(pts.begin()+2, pts.end());
  std::sort(top.begin(), top.end(),
            [](auto&a, auto&b){ return a.x < b.x; });
  std::sort(bot.begin(), bot.end(),
            [](auto&a, auto&b){ return a.x > b.x; });

  std::vector<cv::Point2f> src = {
      top[0], top[1], bot[0], bot[1]
  };
  out_box.clear();
  for (auto& p : src) {
    out_box.emplace_back(cvRound(p.x), cvRound(p.y));
  }

  int dstSize = std::max(W,H);
  std::vector<cv::Point2f> dst = {
      {0,0}, {(float)dstSize-1,0},
      {(float)dstSize-1,(float)dstSize-1},
      {0,(float)dstSize-1}
  };
  out_M = cv::getPerspectiveTransform(src, dst);
  cv::warpPerspective(img, out_warped, out_M,
                      cv::Size(dstSize, dstSize));
  return true;
}

void ExtractPPNodelet::extractFoamBoard(
    const cv::Mat& panel,
    cv::Mat& out_foam,
    std::vector<cv::Point>& out_box)
{
  int H = panel.rows, W = panel.cols;
  int fh = H/2, top = (H - fh)/2;
  out_foam = panel(cv::Rect(0, top, W, fh)).clone();
  out_box = {
      {0, top}, {W-1, top},
      {W-1, top+fh-1}, {0, top+fh-1}
  };
}

std::vector<cv::Point> ExtractPPNodelet::transformFoamBoxToOriginal(
    const std::vector<cv::Point>& foam_box,
    const cv::Mat& M)
{
  cv::Mat M_inv = M.inv();
  std::vector<cv::Point> out;
  for (auto& p : foam_box) {
    cv::Mat v = (cv::Mat_<double>(3,1) << p.x, p.y, 1.0);
    cv::Mat r = M_inv * v;
    double ix = r.at<double>(0,0) / r.at<double>(2,0);
    double iy = r.at<double>(1,0) / r.at<double>(2,0);
    out.emplace_back(cvRound(ix), cvRound(iy));
  }
  return out;
}

void ExtractPPNodelet::publishRaw(
    const cv::Mat& img,
    ros::Publisher& pub,
    const ros::Time& t)
{
  cv_bridge::CvImage out;
  out.header.stamp = t;
  out.encoding      = "bgr8";
  out.image         = img;
  pub.publish(out.toImageMsg());
}

}  // namespace runner

PLUGINLIB_EXPORT_CLASS(runner::ExtractPPNodelet, nodelet::Nodelet)
