#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>

namespace runner {

class ExtractPPNodelet : public nodelet::Nodelet {
public:
  ExtractPPNodelet() = default;
  ~ExtractPPNodelet() {
    _shutdown = true;
    if (_proc_thread.joinable()) _proc_thread.join();
  }

private:
  virtual void onInit() override {
    ros::NodeHandle& nh = getMTNodeHandle();
    ros::NodeHandle& pnh = getMTPrivateNodeHandle();

    _pub_result      = nh.advertise<sensor_msgs::CompressedImage>(
        "/panel_detector/result/compressed", 1);
    _pub_light_panel = nh.advertise<sensor_msgs::CompressedImage>(
        "/panel_detector/light_panel/compressed", 1);
    _pub_foam_board  = nh.advertise<sensor_msgs::CompressedImage>(
        "/panel_detector/foam_board/compressed", 1);

    _sub = nh.subscribe(
        "/hk_camera/image_raw/compressed", 1,
        &ExtractPPNodelet::imageCb, this);

    // 相机标定参数
    _camera_matrix = (cv::Mat_<double>(3,3) <<
                          2343.181585, 0.0,         1221.765641,
                      0.0,         2341.245683, 1040.731733,
                      0.0,         0.0,         1.0);
    _dist_coeffs = (cv::Mat_<double>(1,5) <<
                        -0.080789, 0.084471, 0.000261, 0.000737, 0.0);

    _shutdown = false;
    _proc_thread = std::thread(&ExtractPPNodelet::processingLoop, this);
    NODELET_INFO("ExtractPPNodelet initialized.");
  }

  void imageCb(const sensor_msgs::CompressedImageConstPtr& msg) {
    std::lock_guard<std::mutex> lk(_queue_mutex);
    if (_image_queue.size() < 5) _image_queue.push(msg);
  }

  void processingLoop() {
    ros::Rate rate(100);
    while (ros::ok() && !_shutdown) {
      sensor_msgs::CompressedImageConstPtr msg;
      {
        std::lock_guard<std::mutex> lk(_queue_mutex);
        if (!_image_queue.empty()) {
          msg = _image_queue.front();
          _image_queue.pop();
        }
      }
      if (msg) {
        try {
          cv::Mat raw = cv::imdecode(
              cv::Mat(msg->data), cv::IMREAD_COLOR);
          processImage(raw, msg->header.stamp);
        } catch (const std::exception& e) {
          NODELET_ERROR("Processing exception: %s", e.what());
        }
      }
      rate.sleep();
    }
  }

  void processImage(const cv::Mat& img, const ros::Time& stamp) {
    // 去畸变
    cv::Mat undist;
    cv::Size sz = img.size();
    cv::Mat newK;
    cv::Rect roi;
    newK = cv::getOptimalNewCameraMatrix(
        _camera_matrix, _dist_coeffs, sz, 1, sz, &roi);
    cv::undistort(img, undist, _camera_matrix,
                  _dist_coeffs, newK);
    if (roi.area() > 0)
      undist = undist(roi);

    // 检测 Light Panel
    cv::Mat light_panel;
    std::vector<cv::Point> panel_box;
    cv::Mat M;
    bool found = detectLightPanel(undist, light_panel, panel_box, M);
    if (!found) {
      cv::Mat res = undist.clone();
      cv::putText(res, "Light panel not detected",
                  cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX,
                  1.0, cv::Scalar(0,0,255), 2);
      publishCompressed(res, _pub_result, stamp);
      return;
    }

    // 提取 Foam Board
    cv::Mat foam;
    std::vector<cv::Point> foam_box;
    extractFoamBoard(light_panel, foam, foam_box);

    // 将泡沫板坐标变换回原图
    std::vector<cv::Point> foam_box_orig =
        transformFoamBoxToOriginal(foam_box, panel_box, M);

    // 绘制结果
    cv::Mat res = undist.clone();
    cv::drawContours(res, std::vector<std::vector<cv::Point>>{panel_box},
                     -1, cv::Scalar(0,255,0), 2);
    cv::drawContours(res, std::vector<std::vector<cv::Point>>{foam_box_orig},
                     -1, cv::Scalar(0,0,255), 2);
    cv::putText(res, "Light Panel",
                panel_box[0] - cv::Point(0,10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0,255,0), 2);
    cv::putText(res, "Foam Board",
                foam_box_orig[0] - cv::Point(0,10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0,0,255), 2);

    publishCompressed(res, _pub_result, stamp);
    publishCompressed(light_panel, _pub_light_panel, stamp);
    publishCompressed(foam, _pub_foam_board, stamp);
  }

  bool detectLightPanel(const cv::Mat& img,
                        cv::Mat& out_warped,
                        std::vector<cv::Point>& out_box,
                        cv::Mat& out_M) {
    double scale = 0.5;
    cv::Mat small;
    cv::resize(img, small, cv::Size(), scale, scale);
    cv::Mat gray, blur, thresh;
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5,5), 0);
    cv::threshold(blur, thresh, 0, 255,
                  cv::THRESH_BINARY|cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>> ctrs;
    cv::findContours(thresh, ctrs,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (ctrs.empty()) return false;
    auto maxCt = *std::max_element(ctrs.begin(), ctrs.end(),
                                   [](auto&a, auto&b){ return cv::contourArea(a) < cv::contourArea(b); });
    if (cv::contourArea(maxCt) < 300) return false;
    for (auto& p : maxCt) p *= (1.0/scale);

    cv::RotatedRect rect = cv::minAreaRect(maxCt);
    cv::Point2f boxPts[4];
    rect.points(boxPts);
    std::vector<cv::Point2f> pts(boxPts, boxPts+4);

    int W = (int)rect.size.width;
    int H = (int)rect.size.height;
    double ar = (double)std::max(W,H)/std::max(1,std::min(W,H));
    if (std::abs(ar-1.0) > 0.3) return false;

    std::sort(pts.begin(), pts.end(),
              [](auto&a, auto&b){ return a.y < b.y; });
    std::vector<cv::Point2f> top(pts.begin(), pts.begin()+2);
    std::vector<cv::Point2f> bot(pts.begin()+2, pts.end());
    std::sort(top.begin(), top.end(),
              [](auto&a, auto&b){ return a.x < b.x; });
    std::sort(bot.begin(), bot.end(),
              [](auto&a, auto&b){ return a.x > b.x; });
    std::vector<cv::Point2f> src = {top[0], top[1], bot[0], bot[1]};

    int dstSize = std::max(W,H);
    std::vector<cv::Point2f> dst = {
        {0,0}, {(float)dstSize-1,0},
        {(float)dstSize-1,(float)dstSize-1}, {0,(float)dstSize-1}};

    out_M = cv::getPerspectiveTransform(src, dst);
    cv::warpPerspective(img, out_warped, out_M,
                        cv::Size(dstSize,dstSize));
    out_box.clear();
    for (auto&p : src) out_box.push_back(cv::Point(p));
    return true;
  }

  void extractFoamBoard(const cv::Mat& panel,
                        cv::Mat& out_foam,
                        std::vector<cv::Point>& out_box) {
    int H = panel.rows;
    int W = panel.cols;
    int fh = H/2;
    int top = (H - fh)/2;
    out_foam = panel(cv::Rect(0, top, W, fh)).clone();
    out_box = { {0, top}, {W-1, top}, {W-1, top+fh-1}, {0, top+fh-1} };
  }

  std::vector<cv::Point> transformFoamBoxToOriginal(
      const std::vector<cv::Point>& foam_box,
      const std::vector<cv::Point>& panel_box,
      const cv::Mat& M) {
    cv::Mat M_inv = M.inv();
    std::vector<cv::Point> out;
    for (auto&p : foam_box) {
      cv::Mat v = (cv::Mat_<double>(3,1) << p.x, p.y, 1.0);
      cv::Mat r = M_inv * v;
      double x = r.at<double>(0)/r.at<double>(2);
      double y = r.at<double>(1)/r.at<double>(2);
      out.emplace_back((int)x, (int)y);
    }
    return out;
  }

  void publishCompressed(const cv::Mat& img,
                         ros::Publisher& pub,
                         const ros::Time& t) {
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);
    sensor_msgs::CompressedImage out;
    out.header.stamp = t;
    out.format = "jpeg";
    out.data.assign(buf.begin(), buf.end());
    pub.publish(out);
  }

  ros::Subscriber _sub;
  ros::Publisher  _pub_result, _pub_light_panel, _pub_foam_board;
  cv::Mat         _camera_matrix, _dist_coeffs;
  std::thread     _proc_thread;
  std::mutex      _queue_mutex;
  std::queue<sensor_msgs::CompressedImageConstPtr> _image_queue;
  bool            _shutdown;
};

}  // namespace runner

PLUGINLIB_EXPORT_CLASS(runner::ExtractPPNodelet, nodelet::Nodelet)