//
// Created by lsy on 24-12.9
//

#include "unified_hw/control_loop.h"
#include "unified_hw/hardware_interface.h"

namespace control_loop {

HWControlLoop::HWControlLoop(ros::NodeHandle &nh, std::shared_ptr<Unified::UnifiedHW> hardware_interface)
    : nh_(nh), hardwareInterface_(std::move(hardware_interface)), loopRunning_(true) {
  controllerManager_ = std::make_shared<controller_manager::ControllerManager>(hardwareInterface_.get(), nh_);

  int error = 0;
  int threadPriority = 0;
  ros::NodeHandle nhP("~");
  error += static_cast<int>(!nhP.getParam("loop_frequency", loopHz_));
  error += static_cast<int>(!nhP.getParam("cycle_time_error_threshold", cycleTimeErrorThreshold_));
  error += static_cast<int>(!nhP.getParam("thread_priority", threadPriority));
  if (error > 0) {
    std::string error_message =
        "could not retrieve one of the required parameters: loop_hz or cycle_time_error_threshold or thread_priority";
    ROS_ERROR_STREAM(error_message);
    throw std::runtime_error(error_message);
  }

//  hardwareInterface_->setCanBusThreadPriority(threadPriority);
  hardwareInterface_->init(nh, nhP);

  lastTime_ = Clock::now();

  loopThread_ = std::thread([&]() {
    while (loopRunning_) {
      update();
    }
  });
  sched_param sched{.sched_priority = threadPriority};
  if (pthread_setschedparam(loopThread_.native_handle(), SCHED_FIFO, &sched) != 0) {
    ROS_WARN("Failed to set thread's priority (one possible reason could be that the user and the group permissions are not set properly.).");
  }
}

void HWControlLoop::update() {
  const auto currentTime = Clock::now();
  const Duration desiredDuration(1.0 / loopHz_);
  Duration time_span = std::chrono::duration_cast<Duration>(currentTime - lastTime_);
  elapsedTime_ = ros::Duration(time_span.count());
  lastTime_ = currentTime;

  const double cycle_time_error = (elapsedTime_ - ros::Duration(desiredDuration.count())).toSec();
  if (cycle_time_error > cycleTimeErrorThreshold_) {
    ROS_WARN_STREAM("Cycle time exceeded error threshold by: " << cycle_time_error - cycleTimeErrorThreshold_ << "s, "
                                                               << "cycle time: " << elapsedTime_ << "s, "
                                                               << "threshold: " << cycleTimeErrorThreshold_ << "s");
  }

  hardwareInterface_->read(ros::Time::now(), elapsedTime_);
  controllerManager_->update(ros::Time::now(), elapsedTime_);
  hardwareInterface_->write(ros::Time::now(), elapsedTime_);

  const auto sleepTill = currentTime + std::chrono::duration_cast<Clock::duration>(desiredDuration);
  std::this_thread::sleep_until(sleepTill);
}

HWControlLoop::~HWControlLoop() {
  loopRunning_ = false;
//  hardwareInterface_->close();
  if (loopThread_.joinable()) {
    loopThread_.join();
  }
}

}  // namespace control_loop