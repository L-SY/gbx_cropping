//
// Created by zihan on 2022/6/2.
//

#ifndef SRC_HK_CAMERA_INCLUDE_GALAXY_CAMERA_H_
#define SRC_HK_CAMERA_INCLUDE_GALAXY_CAMERA_H_
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
#include <hk_camera/CameraConfig.h>
#include <camera_info_manager/camera_info_manager.h>
#include <dynamic_reconfigure/client.h>
#include <sensor_msgs/TimeReference.h>
#include <rm_msgs/CameraStatus.h>
#include <string>
#include "libMVSapi/MvCameraControl.h"
#include <rm_msgs/EnableImuTrigger.h>
#include <termios.h>
#include <std_msgs/String.h>

namespace hk_camera
{
struct TriggerPacket
{
  uint32_t trigger_counter_;
  ros::Time trigger_time_;
};
class HKCameraNodelet : public nodelet::Nodelet
{
public:
  HKCameraNodelet();
  ~HKCameraNodelet() override;

  void onInit() override;
  static sensor_msgs::Image image_;
  static sensor_msgs::Image image_rect;

private:
  void reconfigCB(CameraConfig& config, uint32_t level);
  void triggerCB(const sensor_msgs::TimeReference::ConstPtr& time_ref);
  void enableTriggerCB(const ros::TimerEvent&);
  void cameraChange(const std_msgs::String);

  ros::NodeHandle nh_;
  static void* dev_handle_;
  static ros::ServiceClient imu_trigger_client_;
  dynamic_reconfigure::Server<CameraConfig>* srv_{};

  boost::shared_ptr<camera_info_manager::CameraInfoManager> info_manager_;
  static std::string camera_name_;
  std::string camera_info_url_, pixel_format_, frame_id_, camera_sn_;
  double frame_rate_;
  int image_width_{}, image_height_{}, image_offset_x_{}, image_offset_y_{}, sleep_time_{};
  double gain_value_{};
  int gamma_selector_{};
  double gamma_value_{};
  bool initialize_flag_ = true;
  bool gain_auto_{};
  bool exposure_auto_{};
  double exposure_value_{};
  double exposure_max_{};
  double exposure_min_{};
  bool white_auto_{};
  int white_selector_{};
  bool stop_grab_{};
  static int width_;
  static unsigned char* img_;
  static image_transport::CameraPublisher pub_;
  static ros::Publisher pub_rect_;
  static sensor_msgs::CameraInfo info_;
  static std::string imu_name_;
  static bool enable_imu_trigger_;
  static bool enable_resolution_;
  static int resolution_ratio_width_;
  static int resolution_ratio_height_;
  static bool device_open_;
  static void fifoWrite(TriggerPacket pkt);
  static bool fifoRead(TriggerPacket& pkt);
  ros::Subscriber trigger_sub_;
  static bool trigger_not_sync_;
  ros::Timer enable_trigger_timer_;
  ros::Time last_trigger_time_;
  static const int FIFO_SIZE;
  static TriggerPacket fifo_[];
  static uint32_t receive_trigger_counter_;
  static int fifo_front_;
  static int fifo_rear_;
  static bool take_photo_;
  static int count_;
  ros::ServiceServer imu_correspondence_service_;
  static void __stdcall onFrameCB(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
  ros::Subscriber camera_change_sub;
};
}  // namespace hk_camera

#endif  // SRC_HK_CAMERA_INCLUDE_GALAXY_CAMERA_H_
