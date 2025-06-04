//
// Created by zihan on 2022/6/2.
//

#include <pluginlib/class_list_macros.h>
#include <hk_camera.h>
#include <utility>
#include <ros/time.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <iostream>
#include <fstream>
#include <string>

namespace hk_camera
{
PLUGINLIB_EXPORT_CLASS(hk_camera::HKCameraNodelet, nodelet::Nodelet)

HKCameraNodelet::HKCameraNodelet()
{
}

void HKCameraNodelet::onInit()
{
  nh_ = this->getPrivateNodeHandle();
  image_transport::ImageTransport it(nh_);
  pub_ = it.advertiseCamera("image_raw", 1);

  nh_.param("camera_frame_id", image_.header.frame_id, std::string("camera_optical_frame"));
  nh_.param("camera_name", camera_name_, std::string("camera"));
  nh_.param("camera_info_url", camera_info_url_, std::string(""));
  nh_.param("image_width", image_width_, 1440);
  nh_.param("image_height", image_height_, 1080);
  nh_.param("image_offset_x", image_offset_x_, 0);
  nh_.param("image_offset_y", image_offset_y_, 0);
  nh_.param("pixel_format", pixel_format_, std::string("bgr8"));
  nh_.param("frame_id", frame_id_, std::string("camera_optical_frame"));
  nh_.param("camera_sn", camera_sn_, std::string(""));
  nh_.param("frame_rate", frame_rate_, 200.0);
  nh_.param("sleep_time", sleep_time_, 0);
  nh_.param("enable_imu_trigger", enable_imu_trigger_, false);
  nh_.param("imu_name", imu_name_, std::string("gimbal_imu"));
  nh_.param("gain_value", gain_value_, 15.0);
  nh_.param("gain_auto", gain_auto_, false);
  nh_.param("gamma_selector", gamma_selector_, 2);
  nh_.param("gamma_value", gamma_value_, 0.5);
  nh_.param("exposure_auto", exposure_auto_, true);
  nh_.param("exposure_value", exposure_value_, 20.0);
  nh_.param("exposure_max", exposure_max_, 3000.0);
  nh_.param("exposure_min", exposure_min_, 20.0);
  nh_.param("white_auto", white_auto_, true);
  nh_.param("white_selector", white_selector_, 0);
  nh_.param("enable_resolution", enable_resolution_, false);
  nh_.param("resolution_ratio_width", resolution_ratio_width_, 1440);
  nh_.param("resolution_ratio_height", resolution_ratio_height_, 1080);
  nh_.param("stop_grab", stop_grab_, false);

  info_manager_.reset(new camera_info_manager::CameraInfoManager(nh_, camera_name_, camera_info_url_));

  // check for default camera info
  if (!info_manager_->isCalibrated())
  {
    info_manager_->setCameraName(camera_name_);
    sensor_msgs::CameraInfo camera_info;
    camera_info.header.frame_id = image_.header.frame_id;
    camera_info.width = image_width_;
    camera_info.height = image_height_;
    info_manager_->setCameraInfo(camera_info);
  }
  ROS_INFO("Starting '%s' at %dx%d", camera_name_.c_str(), image_width_, image_height_);
  info_ = std::move(info_manager_->getCameraInfo());
  info_.header.frame_id = frame_id_;
  image_.header.frame_id = frame_id_;
  image_.height = image_height_;
  image_.width = image_width_;
  image_.step = image_width_ * 3;
  image_.data.resize(image_.height * image_.step);
  image_.encoding = pixel_format_;
  img_ = new unsigned char[image_.height * image_.step];

  camera_matrix_ = (cv::Mat_<double>(3, 3) <<
                        info_.K[0], info_.K[1], info_.K[2],
                    info_.K[3], info_.K[4], info_.K[5],
                    info_.K[6], info_.K[7], info_.K[8]);
  dist_coeffs_ = cv::Mat(info_.D);

  cam_model_rectified_.fromCameraInfo(info_);

  cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat newCameraMatrix = camera_matrix_;  // 这里直接用原始内参，不做视场裁剪
  cv::Size image_size(info_.width, info_.height);
  cv::initUndistortRectifyMap(
      camera_matrix_, dist_coeffs_,
      R, newCameraMatrix,
      image_size, CV_32FC1, map1_, map2_);
  is_rectify_map_ready_ = true;

  MV_CC_DEVICE_INFO_LIST stDeviceList;
  memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  try
  {
    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (nRet != MV_OK)
      throw(nRet);
  }
  catch (int nRet)
  {
    std::cout << "MV_CC_EnumDevices fail! nRet " << std::hex << nRet << std::endl;
    exit(-1);
  }
  assert(stDeviceList.nDeviceNum > 0);

  // Opens the device.
  unsigned int nIndex = 0;
  MVCC_STRINGVALUE dev_sn;
  memset(&dev_sn, 0, sizeof(MVCC_STRINGVALUE));
  ros::Duration(sleep_time_).sleep();
  if (stDeviceList.nDeviceNum > 1)
  {
    for (; nIndex < stDeviceList.nDeviceNum; nIndex++)
    {
      assert(MV_CC_CreateHandle(&dev_handle_, stDeviceList.pDeviceInfo[nIndex]) == MV_OK);
      MV_CC_OpenDevice(dev_handle_);
      MV_CC_GetStringValue(dev_handle_, "DeviceSerialNumber", &dev_sn);
      if (strcmp(dev_sn.chCurValue, (char*)camera_sn_.data()) == 0)
      {
        break;
      }
      else
      {
        MV_CC_DestroyHandle(dev_handle_);
        if (nIndex == stDeviceList.nDeviceNum - 1)
          ROS_INFO("The serial number is false!");
      }
    }
  }
  else
  {
    assert(MV_CC_CreateHandle(&dev_handle_, stDeviceList.pDeviceInfo[nIndex]) == MV_OK);
    assert(MV_CC_OpenDevice(dev_handle_) == MV_OK);
  }

  MvGvspPixelType format;
  if (pixel_format_ == "mono8")
    format = PixelType_Gvsp_Mono8;
  if (pixel_format_ == "mono16")
    format = PixelType_Gvsp_Mono16;
  if (pixel_format_ == "bgra8")
    format = PixelType_Gvsp_BayerBG8;
  if (pixel_format_ == "rgb8")
    format = PixelType_Gvsp_BayerRG8;
  if (pixel_format_ == "bgr8")
    format = PixelType_Gvsp_BayerGB8;
  if (format == 0)
    static_assert(true, "Illegal format");

  assert(MV_CC_SetIntValue(dev_handle_, "Width", image_width_) == MV_OK);
  assert(MV_CC_SetIntValue(dev_handle_, "Height", image_height_) == MV_OK);
  assert(MV_CC_SetIntValue(dev_handle_, "OffsetX", image_offset_x_) == MV_OK);
  assert(MV_CC_SetIntValue(dev_handle_, "OffsetY", image_offset_y_) == MV_OK);

  _MVCC_FLOATVALUE_T frame_rate;
  MV_CC_SetFrameRate(dev_handle_, frame_rate_);
  MV_CC_GetFrameRate(dev_handle_, &frame_rate);
  ROS_INFO("Frame rate is: %f", frame_rate.fCurValue);

  if (enable_imu_trigger_)
  {
    assert(MV_CC_SetEnumValue(dev_handle_, "TriggerMode", 1) == MV_OK);
    assert(MV_CC_SetEnumValue(dev_handle_, "TriggerSource", MV_TRIGGER_SOURCE_LINE2) == MV_OK);
    assert(MV_CC_SetEnumValue(dev_handle_, "TriggerActivation", 2) == MV_OK);

    trigger_sub_ =
        nh_.subscribe("/rm_hw/" + imu_name_ + "/trigger_time", 50, &hk_camera::HKCameraNodelet::triggerCB, this);
  }
  else
  {
    assert(MV_CC_SetEnumValue(dev_handle_, "TriggerMode", 0) == MV_OK);
  }

  MV_CC_RegisterImageCallBackEx(dev_handle_, onFrameCB, this);

  if (MV_CC_StartGrabbing(dev_handle_) == MV_OK)
  {
    ROS_INFO("Stream On.");
  }

  image_transport::ImageTransport it_rect(nh_);
  pub_rect_ = it_rect.advertiseCamera("image_rect", 1);

  ros::NodeHandle p_nh(nh_, "hk_camera_reconfig");
  srv_ = new dynamic_reconfigure::Server<CameraConfig>(p_nh);
  dynamic_reconfigure::Server<CameraConfig>::CallbackType cb =
      boost::bind(&HKCameraNodelet::reconfigCB, this, _1, _2);
  srv_->setCallback(cb);
  if (enable_imu_trigger_)
  {
    imu_trigger_client_ = nh_.serviceClient<rm_msgs::EnableImuTrigger>("imu_trigger");
    rm_msgs::EnableImuTrigger imu_trigger_srv;
    imu_trigger_srv.request.imu_name = imu_name_;
    imu_trigger_srv.request.enable_trigger = true;
    while (!imu_trigger_client_.call(imu_trigger_srv))
    {
      ROS_WARN("Failed to call service enable_imu_trigger. Retry now.");
      ros::Duration(1).sleep();
    }
    if (imu_trigger_srv.response.is_success)
      ROS_INFO("Enable imu %s trigger camera successfully", imu_name_.c_str());
    else
      ROS_ERROR("Failed to enable imu %s trigger camera", imu_name_.c_str());
    enable_trigger_timer_ = nh_.createTimer(ros::Duration(0.5), &HKCameraNodelet::enableTriggerCB, this);
  }

  camera_change_sub = nh_.subscribe("/camera_name", 50, &hk_camera::HKCameraNodelet::cameraChange, this);
}

void HKCameraNodelet::cameraChange(const std_msgs::String camera_change)
{
  if (strcmp(camera_change.data.c_str(), "hk_camera") == 0)
    MV_CC_StartGrabbing(dev_handle_);
  else
    MV_CC_StopGrabbing(dev_handle_);
}

void HKCameraNodelet::triggerCB(const sensor_msgs::TimeReference::ConstPtr& time_ref)
{
  last_trigger_time_ = time_ref->time_ref;
  hk_camera::TriggerPacket pkt;
  pkt.trigger_time_ = time_ref->time_ref;
  pkt.trigger_counter_ = time_ref->header.seq;
  fifoWrite(pkt);
}

void HKCameraNodelet::enableTriggerCB(const ros::TimerEvent&)
{
  if ((ros::Time::now() - last_trigger_time_).toSec() > 1.0)
  {
    ROS_INFO("Try to enable imu %s to trigger camera.", imu_name_.c_str());
    rm_msgs::EnableImuTrigger imu_trigger_srv;
    imu_trigger_srv.request.imu_name = imu_name_;
    imu_trigger_srv.request.enable_trigger = true;
    imu_trigger_client_.call(imu_trigger_srv);
    if (trigger_not_sync_)
      trigger_not_sync_ = false;
  }
}

void HKCameraNodelet::fifoWrite(TriggerPacket pkt)
{
  if (fifo_front_ == (fifo_rear_ + 1) % FIFO_SIZE)
  {
    ROS_WARN("FIFO overflow!");
    return;
  }
  fifo_[fifo_rear_] = pkt;
  fifo_rear_ = (fifo_rear_ + 1) % FIFO_SIZE;
}

bool HKCameraNodelet::fifoRead(TriggerPacket& pkt)
{
  if (fifo_front_ == fifo_rear_)
    return false;
  pkt = fifo_[fifo_front_];
  fifo_front_ = (fifo_front_ + 1) % FIFO_SIZE;
  return true;
}

void HKCameraNodelet::onFrameCB(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
  if (!pFrameInfo)
    return;

  HKCameraNodelet* self = static_cast<HKCameraNodelet*>(pUser);

  ros::Time now = ros::Time::now();
  if (self->enable_imu_trigger_)
  {
    if (!self->trigger_not_sync_)
    {
      TriggerPacket pkt;
      while (!self->fifoRead(pkt))
      {
        ros::Duration(0.001).sleep();
      }
      if (pkt.trigger_counter_ != self->receive_trigger_counter_++)
      {
        ROS_WARN("Trigger not in sync!");
        self->trigger_not_sync_ = true;
      }
      else if ((now - pkt.trigger_time_).toSec() > 0.013)
      {
        ROS_WARN("Trigger not in sync! Maybe imu %s does not actually trigger camera?", self->imu_name_.c_str());
        self->trigger_not_sync_ = true;
      }
      else
      {
        image_.header.stamp = pkt.trigger_time_;
        info_.header.stamp = pkt.trigger_time_;
      }
    }
    if (self->trigger_not_sync_)
    {
      self->fifo_front_ = self->fifo_rear_;
      rm_msgs::EnableImuTrigger imu_trigger_srv;
      imu_trigger_srv.request.imu_name = self->imu_name_;
      imu_trigger_srv.request.enable_trigger = false;
      self->imu_trigger_client_.call(imu_trigger_srv);
      ROS_INFO("Disable imu %s from triggering camera.", self->imu_name_.c_str());
      self->receive_trigger_counter_ = self->fifo_[self->fifo_rear_ - 1].trigger_counter_ + 1;
      return;
    }
  }
  else
  {
    image_.header.stamp = now;
    info_.header.stamp = now;
  }

  MV_CC_PIXEL_CONVERT_PARAM stConvertParam = { 0 };
  stConvertParam.nWidth = pFrameInfo->nWidth;
  stConvertParam.nHeight = pFrameInfo->nHeight;
  stConvertParam.pSrcData = pData;
  stConvertParam.nSrcDataLen = pFrameInfo->nFrameLen;
  stConvertParam.enSrcPixelType = pFrameInfo->enPixelType;
  stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
  stConvertParam.pDstBuffer = self->img_;
  stConvertParam.nDstBufferSize = pFrameInfo->nWidth * pFrameInfo->nHeight * 3;
  MV_CC_ConvertPixelType(self->dev_handle_, &stConvertParam);
  memcpy((char*)(&image_.data[0]), self->img_, image_.step * image_.height);

  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_, "bgr8");
  cv::Mat raw_bgr = cv_ptr->image;

  cv::Mat rectified_bgr;
  if (self->is_rectify_map_ready_)
  {
    cv::remap(raw_bgr, rectified_bgr, self->map1_, self->map2_, cv::INTER_LINEAR);
  }
  else
  {
    rectified_bgr = raw_bgr;
  }

  sensor_msgs::ImagePtr rect_msg =
      cv_bridge::CvImage(image_.header, "bgr8", rectified_bgr).toImageMsg();
  sensor_msgs::CameraInfo rect_info = info_;
  rect_info.header.stamp = image_.header.stamp;

  for (int i = 0; i < 9; i++)
    rect_info.P[i] = self->camera_matrix_.at<double>(i / 3, i % 3);
  for (int i = 9; i < 12; i++)
    rect_info.P[i] = 0.0;

  self->pub_rect_.publish(*rect_msg, rect_info);

   self->pub_.publish(image_, info_);
}

void HKCameraNodelet::reconfigCB(CameraConfig& config, uint32_t level)
{
  (void)level;

  // Launch setting
  if (initialize_flag_)
  {
    config.exposure_auto = exposure_auto_;
    config.exposure_value = exposure_value_;
    config.exposure_max = exposure_max_;
    config.exposure_min = exposure_min_;
    config.gain_auto = gain_auto_;
    config.gain_value = gain_value_;
    config.gamma_selector = gamma_selector_;
    config.gamma_value = gamma_value_;
    config.white_auto = white_auto_;
    config.white_selector = white_selector_;
    config.stop_grab = stop_grab_;
    initialize_flag_ = false;
  }

  // Switch camera
  if (!config.stop_grab)
    MV_CC_StartGrabbing(dev_handle_);
  else
    MV_CC_StopGrabbing(dev_handle_);

  // Exposure
  if (config.exposure_auto)
  {
    _MVCC_FLOATVALUE_T exposure_time;
    assert(MV_CC_SetIntValue(dev_handle_, "AutoExposureTimeLowerLimit", config.exposure_min) == MV_OK);
    assert(MV_CC_SetIntValue(dev_handle_, "AutoExposureTimeUpperLimit", config.exposure_max) == MV_OK);
    assert(MV_CC_SetEnumValue(dev_handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_CONTINUOUS) == MV_OK);
    assert(MV_CC_GetFloatValue(dev_handle_, "ExposureTime", &exposure_time) == MV_OK);
    config.exposure_value = exposure_time.fCurValue;
  }
  else
  {
    assert(MV_CC_SetEnumValue(dev_handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF) == MV_OK);
    assert(MV_CC_SetFloatValue(dev_handle_, "ExposureTime", config.exposure_value) == MV_OK);
  }

  // Gain
  if (config.gain_auto)
  {
    _MVCC_FLOATVALUE_T gain_value;
    assert(MV_CC_SetFloatValue(dev_handle_, "AutoGainLowerLimit", config.gain_min) == MV_OK);
    assert(MV_CC_SetFloatValue(dev_handle_, "AutoGainUpperLimit", config.gain_max) == MV_OK);
    assert(MV_CC_SetEnumValue(dev_handle_, "GainAuto", MV_GAIN_MODE_CONTINUOUS) == MV_OK);
    assert(MV_CC_GetFloatValue(dev_handle_, "Gain", &gain_value) == MV_OK);
    config.gain_value = gain_value.fCurValue;
  }
  else
  {
    _MVCC_FLOATVALUE_T gain_value;
    assert(MV_CC_SetEnumValue(dev_handle_, "GainAuto", MV_GAIN_MODE_OFF) == MV_OK);
    assert(MV_CC_SetFloatValue(dev_handle_, "Gain", config.gain_value) == MV_OK);
    assert(MV_CC_GetFloatValue(dev_handle_, "Gain", &gain_value) == MV_OK);
    config.gain_value = gain_value.fCurValue;
  }

  // White balance
  assert(MV_CC_SetEnumValue(dev_handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_OFF) == MV_OK);
  switch (config.white_selector)
  {
    case 0:
      assert(MV_CC_SetEnumValue(dev_handle_, "BalanceRatioSelector", 0) == MV_OK);
      break;
    case 1:
      assert(MV_CC_SetEnumValue(dev_handle_, "BalanceRatioSelector", 1) == MV_OK);
      break;
    case 2:
      assert(MV_CC_SetEnumValue(dev_handle_, "BalanceRatioSelector", 2) == MV_OK);
      break;
  }

  _MVCC_INTVALUE_T white_value;
  if (config.white_auto)
  {
    assert(MV_CC_SetEnumValue(dev_handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS) == MV_OK);
    assert(MV_CC_GetIntValue(dev_handle_, "BalanceRatio", &white_value) == MV_OK);
    config.white_value = white_value.nCurValue;
  }
  else
  {
    assert(MV_CC_SetEnumValue(dev_handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_OFF) == MV_OK);
    assert(MV_CC_GetIntValue(dev_handle_, "BalanceRatio", &white_value) == MV_OK);
    config.white_value = white_value.nCurValue;
  }

  switch (config.gamma_selector)
  {
    case 0:
      assert(MV_CC_SetBoolValue(dev_handle_, "GammaEnable", true) == MV_OK);
      assert(MV_CC_SetEnumValue(dev_handle_, "GammaSelector", MV_GAMMA_SELECTOR_SRGB) == MV_OK);
      break;
    case 1:
      assert(MV_CC_SetBoolValue(dev_handle_, "GammaEnable", true) == MV_OK);
      assert(MV_CC_SetEnumValue(dev_handle_, "GammaSelector", MV_GAMMA_SELECTOR_USER) == MV_OK);
      assert(MV_CC_SetGamma(dev_handle_, config.gamma_value) == MV_OK);
      break;
    case 2:
      MV_CC_SetBoolValue(dev_handle_, "GammaEnable", false);
      break;
  }

  take_photo_ = config.take_photo;
}

HKCameraNodelet::~HKCameraNodelet()
{
  MV_CC_StopGrabbing(dev_handle_);
  MV_CC_DestroyHandle(dev_handle_);
}

void* HKCameraNodelet::dev_handle_;
unsigned char* HKCameraNodelet::img_;
sensor_msgs::Image HKCameraNodelet::image_;
sensor_msgs::Image HKCameraNodelet::image_rect;
image_transport::CameraPublisher HKCameraNodelet::pub_;
image_transport::CameraPublisher HKCameraNodelet::pub_rect_;
sensor_msgs::CameraInfo HKCameraNodelet::info_;
int HKCameraNodelet::width_{};
std::string HKCameraNodelet::imu_name_;
std::string HKCameraNodelet::camera_name_;
ros::ServiceClient HKCameraNodelet::imu_trigger_client_;
bool HKCameraNodelet::enable_imu_trigger_;
bool HKCameraNodelet::trigger_not_sync_ = false;
const int HKCameraNodelet::FIFO_SIZE = 1023;
int HKCameraNodelet::count_ = 837;
int HKCameraNodelet::fifo_front_ = 0;
int HKCameraNodelet::fifo_rear_ = 0;
bool HKCameraNodelet::device_open_ = true;
bool HKCameraNodelet::take_photo_{};
struct TriggerPacket HKCameraNodelet::fifo_[FIFO_SIZE];
uint32_t HKCameraNodelet::receive_trigger_counter_ = 0;
bool HKCameraNodelet::enable_resolution_ = false;
int HKCameraNodelet::resolution_ratio_width_ = 1440;
int HKCameraNodelet::resolution_ratio_height_ = 1080;

}  // namespace hk_camera
