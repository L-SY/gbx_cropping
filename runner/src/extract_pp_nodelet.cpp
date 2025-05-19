// extract_pp_nodelet.cpp
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>

namespace runner {

class ExtractPPNodelet : public nodelet::Nodelet {
public:
  ExtractPPNodelet() = default;
  ~ExtractPPNodelet() override {
    // 请求退出并等待线程结束
    {
      std::lock_guard<std::mutex> lk(_queue_mutex);
      _shutdown = true;
    }
    if (_proc_thread.joinable()) {
      _proc_thread.join();
    }
  }

private:
  // Nodelet 初始化
  void onInit() override {
    ros::NodeHandle& nh = getMTNodeHandle();

    // 发布器
    _pub_result      = nh.advertise<sensor_msgs::CompressedImage>(
        "/panel_detector/result/compressed", 1);
    _pub_light_panel = nh.advertise<sensor_msgs::CompressedImage>(
        "/panel_detector/light_panel/compressed", 1);
    _pub_foam_board  = nh.advertise<sensor_msgs::CompressedImage>(
        "/panel_detector/foam_board/compressed", 1);

    // 原始图像订阅
    _sub = nh.subscribe("/hk_camera/image_raw", 1,
                        &ExtractPPNodelet::imageCb, this);

    // 相机标定参数
    _camera_matrix = (cv::Mat_<double>(3,3) <<
                          2343.181585, 0.0,         1221.765641,
                      0.0,         2341.245683, 1040.731733,
                      0.0,         0.0,         1.0);
    _dist_coeffs = (cv::Mat_<double>(1,5) <<
                        -0.080789, 0.084471, 0.000261, 0.000737, 0.0);

    // 去畸变映射尚未初始化
    _maps_initialized = false;
    _shutdown = false;

    // 启动处理线程
    _proc_thread = std::thread(&ExtractPPNodelet::processingLoop, this);

    NODELET_INFO("ExtractPPNodelet initialized.");
  }

  // 回调：收到原始图像
  void imageCb(const sensor_msgs::ImageConstPtr& msg) {
    std::lock_guard<std::mutex> lk(_queue_mutex);
    if (_image_queue.size() < 5) {
      _image_queue.push(msg);
    }
  }

  // 处理线程主循环
  void processingLoop() {
    ros::Rate rate(100);
    while (ros::ok()) {
      sensor_msgs::ImageConstPtr msg;
      {
        std::lock_guard<std::mutex> lk(_queue_mutex);
        if (_shutdown) {
          break;
        }
        if (!_image_queue.empty()) {
          msg = _image_queue.front();
          _image_queue.pop();
        }
      }

      if (msg) {
        try {
          // 转 OpenCV Mat（BGR8）
          cv_bridge::CvImageConstPtr cv_ptr =
              cv_bridge::toCvShare(msg, "bgr8");
          cv::Mat raw = cv_ptr->image;

          // 第一次：根据 raw.size() 初始化去畸变映射表
          if (!_maps_initialized) {
            initUndistortMaps(raw.size());
          }

          // 去畸变
          cv::Mat undist;
          cv::remap(raw, undist, _map1, _map2,
                    cv::INTER_LINEAR);

          // 核心处理
          processImage(undist, msg->header.stamp);
        }
        catch (const std::exception& e) {
          NODELET_ERROR("Processing exception: %s", e.what());
        }
      }

      rate.sleep();
    }
  }

  // 初始化去畸变映射表，只执行一次
  void initUndistortMaps(const cv::Size& image_size) {
    cv::initUndistortRectifyMap(
        _camera_matrix, _dist_coeffs, cv::Mat(),
        _camera_matrix, image_size,
        CV_16SC2, _map1, _map2);
    _maps_initialized = true;
    NODELET_INFO("Undistort maps initialized (%d x %d)",
                 image_size.width, image_size.height);
  }

  // 真正的图像处理逻辑
  void processImage(const cv::Mat& img, const ros::Time& stamp) {
    // 1. 检测 Light Panel
    cv::Mat light_panel;
    std::vector<cv::Point> panel_box;
    cv::Mat M;
    bool found = detectLightPanel(img, light_panel, panel_box, M);

    if (!found) {
      // 未检测到时直接发布原图+文字
      cv::Mat res = img.clone();
      cv::putText(res, "Light panel not detected",
                  cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX,
                  1.0, cv::Scalar(0,0,255), 2);
      publishCompressed(res, _pub_result, stamp);
      return;
    }

    // 2. 提取 Foam Board
    cv::Mat foam;
    std::vector<cv::Point> foam_box;
    extractFoamBoard(light_panel, foam, foam_box);

    // 3. 反变换 Foam Box 到原图
    std::vector<cv::Point> foam_box_orig =
        transformFoamBoxToOriginal(foam_box, M);

    // 4. 在原图上绘制并发布
    cv::Mat annotated = img.clone();
    cv::drawContours(annotated,
                     std::vector<std::vector<cv::Point>>{panel_box},
                     -1, cv::Scalar(0,255,0), 2);
    cv::drawContours(annotated,
                     std::vector<std::vector<cv::Point>>{foam_box_orig},
                     -1, cv::Scalar(0,0,255), 2);

    // 文本标签
    cv::putText(annotated, "Light Panel",
                panel_box[0] - cv::Point(0,10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0,255,0), 2);
    cv::putText(annotated, "Foam Board",
                foam_box_orig[0] - cv::Point(0,10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0,0,255), 2);

    publishCompressed(annotated,     _pub_result,      stamp);
    publishCompressed(light_panel,   _pub_light_panel, stamp);
    publishCompressed(foam,          _pub_foam_board,  stamp);
  }

  // Detect light panel: 返回透视后小图、四点框 & 变换矩阵
  bool detectLightPanel(const cv::Mat& img,
                        cv::Mat& out_warped,
                        std::vector<cv::Point>& out_box,
                        cv::Mat& out_M)
  {
    // 缩放 -> 灰度 -> 二值 -> 找最大轮廓
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

    // 最大区域
    auto max_it = std::max_element(
        ctrs.begin(), ctrs.end(),
        [](auto&a, auto&b){
          return cv::contourArea(a) < cv::contourArea(b);
        });
    if (cv::contourArea(*max_it) < 300) return false;

    // 缩放回原尺寸
    std::vector<cv::Point> contour;
    for (auto& p : *max_it) {
      contour.emplace_back(
          cv::Point(cvRound(p.x/scale),
                    cvRound(p.y/scale)));
    }

    // 最小外接矩形
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f ptsf[4];
    rect.points(ptsf);
    std::vector<cv::Point2f> pts(ptsf, ptsf+4);

    // 长宽比筛选
    int W = cvRound(rect.size.width);
    int H = cvRound(rect.size.height);
    double ar = double(std::max(W,H)) /
                double(std::max(1,std::min(W,H)));
    if (std::abs(ar - 1.0) > 0.3) return false;

    // 角点排序：top-left, top-right, bottom-right, bottom-left
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
      out_box.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
    }

    // 透视变换
    int dstSize = std::max(W,H);
    std::vector<cv::Point2f> dst = {
        {0,0}, {(float)dstSize-1,0},
        {(float)dstSize-1,(float)dstSize-1},
        {0,(float)dstSize-1}
    };
    out_M = cv::getPerspectiveTransform(src, dst);
    cv::warpPerspective(
        img, out_warped, out_M,
        cv::Size(dstSize,dstSize)
    );
    return true;
  }

  // 从透视图里提取中间一半区域（Foam Board）
  void extractFoamBoard(const cv::Mat& panel,
                        cv::Mat& out_foam,
                        std::vector<cv::Point>& out_box)
  {
    int H = panel.rows;
    int W = panel.cols;
    int fh = H / 2;
    int top = (H - fh) / 2;
    out_foam = panel(cv::Rect(0, top, W, fh)).clone();
    out_box = {
        {0, top}, {W-1, top},
        {W-1, top+fh-1}, {0, top+fh-1}
    };
  }

  // 反向透视坐标变换
  std::vector<cv::Point> transformFoamBoxToOriginal(
      const std::vector<cv::Point>& foam_box,
      const cv::Mat& M)
  {
    cv::Mat M_inv = M.inv();
    std::vector<cv::Point> out;
    for (auto& p : foam_box) {
      cv::Mat v = (cv::Mat_<double>(3,1)
                       << p.x, p.y, 1.0);
      cv::Mat r = M_inv * v;
      double ix = r.at<double>(0,0) / r.at<double>(2,0);
      double iy = r.at<double>(1,0) / r.at<double>(2,0);
      out.emplace_back(cv::Point(cvRound(ix), cvRound(iy)));
    }
    return out;
  }

  // 发布压缩图
  void publishCompressed(const cv::Mat& img,
                         ros::Publisher& pub,
                         const ros::Time& t)
  {
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);
    sensor_msgs::CompressedImage out;
    out.header.stamp = t;
    out.format = "jpeg";
    out.data.assign(buf.begin(), buf.end());
    pub.publish(out);
  }

  // 成员变量
  ros::Subscriber                                     _sub;
  ros::Publisher                                      _pub_result, _pub_light_panel, _pub_foam_board;
  cv::Mat                                             _camera_matrix, _dist_coeffs;
  cv::Mat                                             _map1, _map2;
  bool                                                _maps_initialized = false;
  std::thread                                         _proc_thread;
  std::mutex                                          _queue_mutex;
  std::queue<sensor_msgs::ImageConstPtr>              _image_queue;
  bool                                                _shutdown = false;
};

}  // namespace runner

PLUGINLIB_EXPORT_CLASS(runner::ExtractPPNodelet, nodelet::Nodelet)
