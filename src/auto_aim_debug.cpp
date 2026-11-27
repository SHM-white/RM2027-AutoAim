#include <Eigen/Geometry>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <list>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/recorder.hpp"
#include "detection_timing_panel.hpp"

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | yaml 配置文件路径 }"
  "{enable-control | true                | 开启云台跟随，默认开启 }"
  "{detect-timing  | true                | 显示 detect 分段耗时及最近 60 帧统计 }"
  "{headless       | false               | 无头模式：不创建任何窗口，终端约每秒刷新一次帧率 }"
  "{allow-unlit    | false               | 允许灯条未亮（extinguish）的装甲板参与跟踪、预测与瞄准 }"
  "{record         | false               | 保存视频与对应云台四元数到 records/（仅记录云台姿态有效的帧）}"
  "{record-fps     | 30                  | 录像帧率上限，同时作为保存视频的帧率 }";

namespace
{
constexpr double degrees = 180.0 / CV_PI;
const cv::Scalar foreground(225, 225, 230);
const cv::Scalar muted(145, 150, 160);
const cv::Scalar detection_color(255, 180, 70);
const cv::Scalar tracked_color(90, 225, 100);
const cv::Scalar feedback_color(255, 220, 80);
const cv::Scalar reference_color(0, 220, 255);
const cv::Scalar command_color(230, 80, 230);

void text(cv::Mat & image, const std::string & value, int x, int y,
          const cv::Scalar & color = foreground, double scale = 0.52)
{
  cv::putText(image, value, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

Eigen::Quaterniond from_ypr(double yaw, double pitch, double roll = 0)
{
  return Eigen::Quaterniond(
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
}

std::optional<Eigen::Vector3d> eulers(const Eigen::Quaterniond & q)
{
  if (!q.coeffs().allFinite() || q.norm() < 1e-6) return std::nullopt;
  const Eigen::Matrix3d r = q.normalized().toRotationMatrix();
  return Eigen::Vector3d{
    std::atan2(r(1, 0), r(0, 0)), std::asin(std::clamp(-r(2, 0), -1.0, 1.0)),
    std::atan2(r(2, 1), r(2, 2))};
}

void draw_pose(cv::Mat & image, int y, const std::string & title,
               const std::optional<Eigen::Quaterniond> & q, const cv::Scalar & color)
{
  constexpr int x = 990;
  text(image, title, x, y, color, 0.55);
  if (!q) {
    text(image, "N/A", x, y + 24, muted);
    return;
  }
  text(image, fmt::format("wxyz {:+.5f} {:+.5f} {:+.5f} {:+.5f}",
    q->w(), q->x(), q->y(), q->z()), x, y + 24);
  const auto ypr = eulers(*q);
  if (ypr)
    text(image, fmt::format("YPR {:+.2f} / {:+.2f} / {:+.2f} deg",
      (*ypr)[0] * degrees, (*ypr)[1] * degrees, (*ypr)[2] * degrees), x, y + 48);
  else
    text(image, "INVALID QUATERNION: Euler / direction unavailable", x, y + 48,
      cv::Scalar(70, 70, 255));
}

struct AimView
{
  io::Command command{false, false, 0, 0};
  std::optional<Eigen::Vector4d> aim_xyza;
  Clock::time_point source_time{};
  Clock::time_point computed{};
  double solve_ms = 0;
  bool tx_fire_request = false;
  std::string status = "Waiting for live image and pose";
  std::string control_status = "RX ONLY - no serial TX";
};

// Owns the aimer. The GUI receives copies under one mutex.
// The optional gimbal is the only command writer. A null pointer preserves RX-only mode.
class AimWorker
{
public:
  AimWorker(const std::string & config_path, io::Gimbal * control_gimbal)
  : aimer_(config_path),
    shooter_(config_path),
    control_gimbal_(control_gimbal),
    thread_([this] { run(); })
  {}

  ~AimWorker() { stop(); }

  void stop()
  {
    quit_ = true;
    if (thread_.joinable()) thread_.join();
  }

  void submit(const std::optional<auto_aim::Target> & target, Clock::time_point t,
              bool pose_valid, bool tracking_ready, double bullet_speed, uint64_t generation,
              const Eigen::Vector3d & gimbal_pos, bool gimbal_pos_valid)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool same_target = input_.target && target &&
      input_.target->name == target->name && input_.target->armor_type == target->armor_type;
    const bool ready = pose_valid && tracking_ready && target.has_value();
    if (!ready || !input_.pose_valid || !input_.tracking_ready || !same_target ||
        input_.generation != generation)
      ++input_.epoch;
    input_ = {target,          t,           pose_valid,       tracking_ready,
              bullet_speed,    generation,  gimbal_pos,       gimbal_pos_valid,
              input_.epoch};
  }

  AimView view()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return output_;
  }

private:
  struct Input
  {
    std::optional<auto_aim::Target> target;
    Clock::time_point t{};
    bool pose_valid = false;
    bool tracking_ready = false;
    double bullet_speed = 0;
    uint64_t generation = 0;
    Eigen::Vector3d gimbal_pos = Eigen::Vector3d::Zero();
    bool gimbal_pos_valid = false;
    uint64_t epoch = 0;
  };

  void transmit(const Input & input, AimView & output)
  {
    if (!control_gimbal_) return;
    const auto now = Clock::now();
    const auto feedback = control_gimbal_->feedback();
    bool current_input;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_input = input.epoch == input_.epoch;
    }
    io::Command tx{false, false, 0, 0};
    if (quit_)
      output.control_status = "TX STOP: exiting";
    else if (!feedback || now - feedback->received > 100ms)
      output.control_status = "TX STOP: feedback missing / stale";
    else if (feedback->mode != io::GimbalMode::AUTO_AIM)
      output.control_status = "TX STOP: board mode is not AUTO_AIM";
    else if (feedback->generation != input.generation || !current_input)
      output.control_status = "TX STOP: source changed / reconnected";
    else if (!input.pose_valid || input.t == Clock::time_point{} || now - input.t > 250ms)
      output.control_status = "TX STOP: image / pose missing or stale";
    else if (!input.target || !input.tracking_ready)
      output.control_status = "TX STOP: target is not TRACKING";
    else if (!output.command.control ||
             !std::isfinite(output.command.yaw) || !std::isfinite(output.command.pitch))
      output.control_status = "TX STOP: no valid aim command";
    else {
      tx = output.command;
      output.control_status = "TX FOLLOW";
    }
    output.tx_fire_request = tx.control && tx.shoot;
    // Ordinary (non-MPC) control only commands angles; velocity / acceleration are zero.
    control_gimbal_->send(
      tx.control, tx.shoot, static_cast<float>(tx.yaw), 0, 0,
      static_cast<float>(tx.pitch), 0, 0);
  }

  void run()
  {
    std::string last_control_status;
    while (!quit_) {
      const auto start = Clock::now();
      Input input;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        input = input_;
      }
      AimView output;
      output.source_time = input.t;
      if (!input.pose_valid)
        output.status = "No image-matched pose: aiming paused";
      else if (start - input.t > 250ms)
        output.status = "Image / target older than 250 ms: aiming paused";
      else if (!input.target)
        output.status = "No tracked target";
      else {
        try {
          const std::list<auto_aim::Target> targets{*input.target};
          const auto command = aimer_.aim(targets, input.t, input.bullet_speed);
          if (command.control && std::isfinite(command.yaw) && std::isfinite(command.pitch)) {
            output.command = command;
            if (input.gimbal_pos_valid)
              output.command.shoot =
                shooter_.shoot(output.command, aimer_, targets, input.gimbal_pos);
            if (aimer_.debug_aim_point.valid)
              output.aim_xyza = aimer_.debug_aim_point.xyza;
            output.status = "Live aim result";
          } else {
            output.command = {false, false, 0, 0};
            output.aim_xyza.reset();
            output.status = "No valid ballistic / aim solution";
          }
        } catch (const std::exception & e) {
          output.command = {false, false, 0, 0};
          output.aim_xyza.reset();
          output.status = std::string("Aimer: ") + e.what();
        }
      }
      output.computed = Clock::now();
      output.solve_ms = std::chrono::duration<double, std::milli>(output.computed - start).count();
      transmit(input, output);
      if (control_gimbal_ && output.control_status != last_control_status) {
        tools::logger()->info("[AutoAim Debug] {}", output.control_status);
        last_control_status = output.control_status;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        output_ = std::move(output);
      }
      std::this_thread::sleep_until(start + 10ms);
    }
    if (control_gimbal_) {
      control_gimbal_->send(false, false, 0, 0, 0, 0, 0, 0);
      tools::logger()->info("[AutoAim Debug] 控制线程退出，已请求停控、不开火");
    }
  }

  auto_aim::Aimer aimer_;
  auto_aim::Shooter shooter_;
  io::Gimbal * control_gimbal_;
  std::mutex mutex_;
  Input input_;
  AimView output_;
  std::atomic<bool> quit_ = false;
  std::thread thread_;
};

class Projection
{
public:
  explicit Projection(const YAML::Node & yaml)
  {
    const auto k = yaml["camera_matrix"].as<std::vector<double>>();
    const auto d = yaml["distort_coeffs"].as<std::vector<double>>();
    const auto r = yaml["R_camera2gimbal"].as<std::vector<double>>();
    const auto t = yaml["t_camera2gimbal"].as<std::vector<double>>();
    const auto imu = yaml["R_gimbal2imubody"].as<std::vector<double>>();
    if (k.size() != 9 || r.size() != 9 || t.size() != 3 || imu.size() != 9 || d.empty())
      throw std::runtime_error("Invalid camera / gimbal calibration dimensions");
    camera_matrix_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(k.data())).clone();
    distortion_ = cv::Mat(d).clone();
    camera2gimbal_ = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r.data());
    translation_ = Eigen::Map<const Eigen::Vector3d>(t.data());
    gimbal2imubody_ = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(imu.data());
  }

  Eigen::Quaterniond raw_in_gimbal_world(const Eigen::Quaterniond & q) const
  {
    return Eigen::Quaterniond(
      gimbal2imubody_.transpose() * q.normalized().toRotationMatrix() * gimbal2imubody_);
  }

  std::vector<std::optional<cv::Point>> pixels(
    const std::vector<Eigen::Vector3d> & world, const Eigen::Matrix3d & gimbal2world,
    const cv::Point2d & display_scale) const
  {
    std::vector<std::optional<cv::Point>> result(world.size());
    std::vector<cv::Point3d> camera_points;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < world.size(); ++i) {
      const Eigen::Vector3d p = camera2gimbal_.transpose() *
        (gimbal2world.transpose() * world[i] - translation_);
      if (!p.allFinite() || p.z() <= 0.01) continue;
      camera_points.emplace_back(p.x(), p.y(), p.z());
      indices.push_back(i);
    }
    if (camera_points.empty()) return result;
    std::vector<cv::Point2d> image_points;
    cv::projectPoints(camera_points, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
      camera_matrix_, distortion_, image_points);
    for (std::size_t i = 0; i < image_points.size(); ++i) {
      const auto & p = image_points[i];
      if (std::isfinite(p.x) && std::isfinite(p.y) &&
          std::abs(p.x) < 1e6 && std::abs(p.y) < 1e6)
        result[indices[i]] = cv::Point(cvRound(p.x * display_scale.x), cvRound(p.y * display_scale.y));
    }
    return result;
  }

private:
  cv::Mat camera_matrix_, distortion_;
  Eigen::Matrix3d camera2gimbal_, gimbal2imubody_;
  Eigen::Vector3d translation_;
};

void draw_path(cv::Mat & image, const std::vector<std::optional<cv::Point>> & points,
               const cv::Scalar & color, int thickness = 2)
{
  for (std::size_t i = 1; i < points.size(); ++i) {
    if (!points[i - 1] || !points[i]) continue;
    auto a = *points[i - 1], b = *points[i];
    if (cv::clipLine(image.size(), a, b)) cv::line(image, a, b, color, thickness, cv::LINE_AA);
  }
}

void draw_polygon(cv::Mat & image, const std::vector<cv::Point2f> & points,
                  const cv::Scalar & color, const cv::Point2d & display_scale)
{
  if (points.empty()) return;
  std::vector<std::optional<cv::Point>> polygon;
  polygon.reserve(points.size() + 1);
  for (const auto & p : points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
        std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6) return;
    polygon.emplace_back(cv::Point(cvRound(p.x * display_scale.x), cvRound(p.y * display_scale.y)));
  }
  polygon.push_back(polygon.front());
  draw_path(image, polygon, color);
}

void draw_camera(
  cv::Mat & image, const cv::Size & source_size, const std::list<auto_aim::Armor> & detections,
  const std::optional<auto_aim::Target> & target, auto_aim::Solver & solver,
  const Projection & projection, const AimView & view, bool pose_valid)
{
  const cv::Point2d display_scale(
    static_cast<double>(image.cols) / source_size.width,
    static_cast<double>(image.rows) / source_size.height);
  for (const auto & armor : detections) {
    draw_polygon(image, armor.points, detection_color, display_scale);
    const cv::Point origin(
      std::clamp(cvRound(armor.center.x * display_scale.x), 0, image.cols - 1),
      std::clamp(cvRound(armor.center.y * display_scale.y), std::min(18, image.rows - 1), image.rows - 1));
    text(image, fmt::format("{} {} {:.2f}", auto_aim::COLORS.at(armor.color),
      auto_aim::ARMOR_NAMES.at(armor.name), armor.confidence), origin.x, origin.y,
      detection_color, 0.6);
  }
  if (!pose_valid || !target) return;
  const Eigen::Matrix3d rotation = solver.R_gimbal2world();
  for (const auto & xyza : target->armor_xyza_list()) {
    if (!projection.pixels({xyza.head<3>()}, rotation, display_scale).front()) continue;
    draw_polygon(image,
      solver.reproject_armor(xyza.head<3>(), xyza[3], target->armor_type, target->name),
      tracked_color, display_scale);
  }
  if (!view.command.control) return;
  const double distance = view.aim_xyza
    ? std::max(0.2, view.aim_xyza->head<3>().norm()) : 1.0;
  if (view.aim_xyza) {
    const auto aim = projection.pixels({view.aim_xyza->head<3>()}, rotation, display_scale);
    if (aim.front() && cv::Rect(0, 0, image.cols, image.rows).contains(*aim.front()))
      cv::drawMarker(image, *aim.front(), reference_color, cv::MARKER_DIAMOND, 22, 2);
  }
  const Eigen::Vector3d current = rotation * Eigen::Vector3d::UnitX() * distance;
  const Eigen::Vector3d command =
    from_ypr(view.command.yaw, view.command.pitch) * Eigen::Vector3d::UnitX() * distance;
  const auto direction = projection.pixels({current, command}, rotation, display_scale);
  if (direction[0] && direction[1]) {
    auto a = *direction[0], b = *direction[1];
    if (cv::clipLine(image.size(), a, b))
      cv::arrowedLine(image, a, b, command_color, 3, cv::LINE_AA, 0, 0.15);
  }
  if (direction[1] && cv::Rect(0, 0, image.cols, image.rows).contains(*direction[1]))
    cv::drawMarker(image, *direction[1], command_color, cv::MARKER_CROSS, 26, 2);
}

void draw_directions(cv::Mat & canvas, const std::optional<Eigen::Quaterniond> & received,
                     const std::optional<Eigen::Quaterniond> & command,
                     const std::optional<Eigen::Quaterniond> & armor, const std::string & control_status)
{
  text(canvas, "Directions in world | view from +Z side", 990, 680, foreground, 0.55);
  const cv::Point origin(1260, 840);
  const auto project = [&](const Eigen::Vector3d & p) {
    return origin + cv::Point(cvRound(118 * (p.y() - p.x()) / std::sqrt(2.0)),
      cvRound(118 * ((p.x() + p.y()) / std::sqrt(8.0) - 0.8660254 * p.z())));
  };
  for (int axis = 0; axis < 3; ++axis) {
    const auto end = project(Eigen::Vector3d::Unit(axis) * 1.35);
    cv::arrowedLine(canvas, origin, end, muted, 1, cv::LINE_AA);
    text(canvas, std::string("W") + "XYZ"[axis], end.x, end.y, muted);
  }
  const auto arrow = [&](const std::optional<Eigen::Quaterniond> & q,
                         const cv::Scalar & color, const std::string & label, double length) {
    if (!q || !eulers(*q)) return;
    const auto end = project(*q * Eigen::Vector3d::UnitX() * length);
    cv::arrowedLine(canvas, origin, end, color, 3, cv::LINE_AA, 0, 0.13);
    text(canvas, label, end.x + 5, end.y - 5, color, 0.48);
  };
  arrow(received, feedback_color, "RX forward", 1.10);
  arrow(command, command_color, "CMD forward", 1.30);
  arrow(armor, tracked_color, "armor normal", 0.85);
  text(canvas, "Cyan RX | Magenta command | Green armor normal", 990, 963, muted, 0.48);
  text(canvas, control_status, 990, 986, command_color, 0.5);
}

int run(const std::string & config_path, bool enable_control, bool detect_timing, bool headless,
        bool allow_unlit, bool record, double record_fps)
{
  const auto yaml = YAML::LoadFile(config_path);
  const double default_speed = yaml["default_bullet_speed"].as<double>();
  if (!std::isfinite(default_speed) || default_speed < 10 || default_speed > 25)
    throw std::runtime_error("default_bullet_speed must be within 10..25 m/s");
  const Projection projection(yaml);
  tools::Exiter exiter;
  // Only the opt-in aim worker can send; no IMU forwarding or navigation callbacks.
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path, enable_control
    ? io::CameraFramePolicy::keep_queued : io::CameraFramePolicy::keep_latest);
  auto_aim::YOLO yolo(config_path, false);
  yolo.set_timing_enabled(detect_timing);
  DetectionTimingPanel timing_panel(detect_timing && !headless);
  auto_aim::Solver solver(config_path);
  auto tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
  tracker->set_allow_unlit(allow_unlit);
  AimWorker worker(config_path, enable_control ? &gimbal : nullptr);
  std::optional<tools::Recorder> recorder;
  if (record) {
    recorder.emplace(record_fps);
    tools::logger()->info("录像已开启：records/ 下保存 .avi 与逐帧云台四元数 .txt，帧率上限 {} Hz",
      record_fps);
  }
  tools::logger()->info("灯条未亮（extinguish）的装甲板：{}",
    allow_unlit ? "允许跟踪 / 预测 / 瞄准" : "仍按敌方颜色过滤");
  tools::logger()->info("普通自瞄实机可视化（非 MPC）：{}", headless
    ? "无头模式，不创建窗口，终端约每秒刷新帧率"
    : (enable_control ? "已启用云台跟随" : "仅接收串口反馈，不发送任何控制/开火帧"));
  tools::logger()->info("弹速反馈不可用时读取配置默认值 {} m/s", default_speed);
  if (detect_timing && !headless)
    tools::logger()->info("detect 分段计时已开启：T 显示/隐藏计时窗口，R 清除最近 60 帧统计");

  const std::string window = enable_control
    ? "Auto aim (direct) - FOLLOW" : "Auto aim (direct) - RX ONLY";
  if (!headless) {
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 1280, 800);
  }
  const cv::Size canvas_size(1600, 1000);
  const cv::Scalar canvas_background(24, 25, 30);
  cv::Mat preview_canvas;
  if (!headless && !enable_control) preview_canvas.create(canvas_size, CV_8UC3);
  cv::Mat image;
  Clock::time_point image_time;
  int frame_count = 0;
  int fps_frames = 0;
  auto fps_window_start = Clock::now();
  bool had_pose = false;
  bool window_query_enabled = true;
  bool window_observed = false;
  bool window_closed = false;
  std::string exit_reason = "SIGINT / Ctrl+C";
  auto previous_frame = Clock::now();

  while (!exiter.exit()) {
    camera.read(image, image_time);
    const auto frame_read = Clock::now();
    if (image.empty()) continue;
    const auto image_q = gimbal.try_q(image_time);
    const auto feedback = gimbal.feedback();
    const bool pose_valid = image_q.has_value() && feedback.has_value();
    if (recorder && pose_valid && image_q) recorder->record(image, *image_q, image_time);
    const auto detection_start = Clock::now();
    const auto detections = yolo.detect(image, frame_count++);
    const auto tracking_start = Clock::now();
    std::optional<auto_aim::Target> target;
    std::optional<Eigen::Quaterniond> armor_q, pipeline_q;
    if (pose_valid) {
      solver.set_R_gimbal2world(*image_q);
      pipeline_q = Eigen::Quaterniond(solver.R_gimbal2world());
      auto armors = detections;
      const auto targets = tracker->track(armors, image_time);
      if (!targets.empty()) {
        target = targets.front();
        const auto observed = std::find_if(armors.begin(), armors.end(), [&](const auto & armor) {
          return armor.name == target->name && armor.type == target->armor_type;
        });
        if (observed != armors.end() && observed->ypr_in_world.allFinite()) {
          const auto & ypr = observed->ypr_in_world;
          armor_q = from_ypr(ypr[0], ypr[1], ypr[2]);
        }
      }
    } else if (had_pose) {
      tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
      tracker->set_allow_unlit(allow_unlit);
    }
    had_pose = pose_valid;
    const double board_speed = feedback ? feedback->state.bullet_speed : 0;
    const bool use_default = !std::isfinite(board_speed) || board_speed < 10 || board_speed > 25;
    const double speed = use_default ? default_speed : board_speed;
    const auto gimbal_eulers = feedback ? eulers(feedback->q) : std::nullopt;
    worker.submit(target, image_time, pose_valid, tracker->state() == "tracking", speed,
      feedback ? feedback->generation : 0,
      gimbal_eulers ? *gimbal_eulers : Eigen::Vector3d::Zero(), gimbal_eulers.has_value());
    const auto tracking_finish = Clock::now();
    auto view = worker.view();
    if (!pose_valid || !target || view.source_time == Clock::time_point{} ||
        Clock::now() - view.source_time > 250ms) {
      view.command = {false, false, 0, 0};
      view.aim_xyza.reset();
      view.status = !pose_valid ? "No image-matched pose: aiming paused" :
        (!target ? "No tracked target" : "Waiting for a fresh aim result");
    }
    std::optional<Eigen::Quaterniond> raw_q, raw_direction, command_q;
    if (feedback) {
      raw_q = feedback->q;
      if (eulers(*raw_q)) raw_direction = projection.raw_in_gimbal_world(*raw_q);
    }
    if (view.command.control)
      command_q = from_ypr(view.command.yaw, view.command.pitch);

    if (headless) {
      ++fps_frames;
      const auto fps_now = Clock::now();
      const double elapsed = std::chrono::duration<double>(fps_now - fps_window_start).count();
      if (elapsed >= 1.0) {
        const double detection_ms = std::chrono::duration<double, std::milli>(
          tracking_start - detection_start).count();
        const double tracker_ms = std::chrono::duration<double, std::milli>(
          tracking_finish - tracking_start).count();
        const double image_age = std::chrono::duration<double, std::milli>(
          fps_now - image_time).count();
        tools::logger()->info("FPS {:.1f} | detect {:.1f} ms | track {:.1f} ms | image age {:.1f} ms",
          fps_frames / elapsed, detection_ms, tracker_ms, image_age);
        fps_frames = 0;
        fps_window_start = fps_now;
      }
      continue;
    }

    cv::Mat canvas = enable_control ? cv::Mat(canvas_size, CV_8UC3) : preview_canvas;
    canvas.setTo(canvas_background);
    text(canvas, fmt::format("AUTO AIM (DIRECT / NON-MPC): CAMERA -> DETECTION -> TRACKER -> AIMER | {} | q / Esc exit",
      enable_control ? "FOLLOW" : "RX ONLY"),
      16, 29, command_color, 0.65);
    const double scale = std::min(960.0 / image.cols, 720.0 / image.rows);
    const cv::Size preview_size(
      std::max(1, cvRound(image.cols * scale)), std::max(1, cvRound(image.rows * scale)));
    const cv::Rect preview_area(10 + (960 - preview_size.width) / 2,
      48 + (720 - preview_size.height) / 2, preview_size.width, preview_size.height);
    cv::Mat preview = canvas(preview_area);
    cv::Mat drawing = enable_control ? image.clone() : preview;
    if (!enable_control) cv::resize(image, drawing, preview_size, 0, 0, cv::INTER_LINEAR);
    draw_camera(drawing, image.size(), detections, target, solver, projection, view, pose_valid);
    if (enable_control) {
      cv::Mat resized;
      cv::resize(drawing, resized, preview_size, 0, 0, cv::INTER_AREA);
      resized.copyTo(preview);
    }
    const auto now = Clock::now();
    const double frame_seconds = std::chrono::duration<double>(now - previous_frame).count();
    previous_frame = now;
    const double detection_ms = std::chrono::duration<double, std::milli>(tracking_start - detection_start).count();
    const double tracker_ms = std::chrono::duration<double, std::milli>(tracking_finish - tracking_start).count();
    const double image_age = std::chrono::duration<double, std::milli>(now - image_time).count();
    const double read_age = std::chrono::duration<double, std::milli>(frame_read - image_time).count();
    text(canvas, fmt::format("{:.1f} FPS | detect {:.1f} ms | track {:.1f} ms | read age {:.1f} ms | image age {:.1f} ms",
      frame_seconds > 0 ? 1 / frame_seconds : 0, detection_ms, tracker_ms, read_age, image_age), 22, 73,
      cv::Scalar(255, 255, 255), 0.55);
    const auto & detect_profile = yolo.timing();
    if (detect_timing && detect_profile.valid) {
      text(canvas, fmt::format("pre {:.2f} | forward {:.2f} | post {:.2f} | other {:.2f} ms | T timing / R reset",
        detect_profile.preprocessing_ms(), detect_profile.ms(auto_aim::DetectStage::forward),
        detect_profile.postprocessing_ms(), detect_profile.ms(auto_aim::DetectStage::other)),
        22, 96, cv::Scalar(255, 255, 255), 0.49);
      timing_panel.add(detect_profile);
    }
    text(canvas, "Blue detection | Green track | Yellow aim point | Magenta direct command", 18, 795,
      foreground, 0.53);
    text(canvas, fmt::format("TX fire request: {}", view.tx_fire_request ? "YES" : "NO"), 18, 768,
      view.tx_fire_request ? command_color : muted, 0.58);
    text(canvas, fmt::format("Feedback: {} | matched pose: {} | mode: {}", feedback ? "LIVE" : "MISSING",
      pose_valid ? "YES" : "NO", feedback ? gimbal.str(feedback->mode) : "N/A"), 990, 58,
      pose_valid ? tracked_color : reference_color, 0.52);

    draw_pose(canvas, 90, "RECEIVED RAW q (latest AB packet, body -> world)", raw_q, feedback_color);
    if (feedback)
      text(canvas, fmt::format("|q| {:.6f} | age {:.1f} ms | packet Y/P {:+.2f}/{:+.2f}",
        feedback->q.norm(), std::chrono::duration<double, std::milli>(now - feedback->received).count(),
        feedback->state.yaw * degrees, feedback->state.pitch * degrees), 990, 162, muted, 0.49);
    draw_pose(canvas, 192, "PIPELINE @ IMAGE (yaw/pitch source, roll = 0)", pipeline_q, foreground);
    draw_pose(canvas, 282, "DESIRED GIMBAL q (direct command; roll = 0)", command_q, command_color);
    if (command_q && feedback) {
      text(canvas, fmt::format("CMD - packet Y/P: {:+.2f} / {:+.2f} deg",
        std::remainder(view.command.yaw - feedback->state.yaw, 2 * CV_PI) * degrees,
        std::remainder(view.command.pitch - feedback->state.pitch, 2 * CV_PI) * degrees), 990, 354);
      text(canvas, fmt::format("CMD yaw/pitch {:+.2f}/{:+.2f} deg | fire {}",
        view.command.yaw * degrees, view.command.pitch * degrees, view.command.shoot ? "YES" : "NO"),
        990, 378, command_color, 0.49);
    }
    draw_pose(canvas, 414, "OBSERVED ARMOR q (PnP + yaw fit, armor -> world)", armor_q, tracked_color);
    if (target) {
      const auto state = target->ekf_x();
      text(canvas, fmt::format("Target {} | tracker {} | detections {}",
        auto_aim::ARMOR_NAMES.at(target->name), tracker->state(), detections.size()), 990, 492);
      text(canvas, fmt::format("Filtered center XYZ {:+.3f} / {:+.3f} / {:+.3f} m",
        state[0], state[2], state[4]), 990, 516);
      text(canvas, fmt::format("Filtered heading {:+.2f} deg | spin {:+.2f} rad/s",
        state[6] * degrees, state[7]), 990, 540);
    } else {
      text(canvas, fmt::format("Target N/A | tracker {} | detections {}",
        pose_valid ? tracker->state() : "WAIT FOR POSE", detections.size()), 990, 492, muted);
    }
    text(canvas, fmt::format("Bullet speed {:.2f} m/s [{}] | board {:.2f}", speed,
      use_default ? "CONFIG DEFAULT" : "BOARD", board_speed), 990, 576,
      use_default ? reference_color : foreground, 0.51);
    if (view.command.control)
      text(canvas, fmt::format("Source age {:.1f} ms | aimer {:.1f} ms",
        std::chrono::duration<double, std::milli>(now - view.source_time).count(),
        view.solve_ms), 990, 600, muted, 0.49);
    text(canvas, view.status.substr(0, 68), 990, 630, command_q ? tracked_color : reference_color, 0.49);
    text(canvas, fmt::format("Configured offsets Y/P {:+.2f} / {:+.2f} deg",
      yaml["yaw_offset"].as<double>(), yaml["pitch_offset"].as<double>()), 990, 654, muted, 0.49);

    cv::rectangle(canvas, {10, 815, 960, 180}, cv::Scalar(65, 65, 75));
    text(canvas, "DIRECT AIM (Aimer, no MPC)", 24, 845, reference_color, 0.6);
    if (view.aim_xyza)
      text(canvas, fmt::format("Aim point world XYZ {:+.3f} / {:+.3f} / {:+.3f} m | armor yaw {:+.2f} deg",
        (*view.aim_xyza)[0], (*view.aim_xyza)[1], (*view.aim_xyza)[2],
        (*view.aim_xyza)[3] * degrees), 24, 877);
    else
      text(canvas, "Aim point N/A", 24, 877, muted);
    text(canvas, fmt::format("Command yaw/pitch {:+.2f} / {:+.2f} deg | shoot {}",
      view.command.yaw * degrees, view.command.pitch * degrees,
      view.command.shoot ? "YES" : "NO"), 24, 905, command_color);
    if (feedback && view.command.control)
      text(canvas, fmt::format("Command - packet Y/P {:+.2f} / {:+.2f} deg",
        std::remainder(view.command.yaw - feedback->state.yaw, 2 * CV_PI) * degrees,
        std::remainder(view.command.pitch - feedback->state.pitch, 2 * CV_PI) * degrees),
        24, 933, feedback_color);
    text(canvas, view.status.substr(0, 100), 24, 963, command_q ? tracked_color : reference_color, 0.5);

    draw_directions(canvas, raw_direction, command_q, armor_q, view.control_status);
    timing_panel.show();
    cv::imshow(window, canvas);
    const int key = cv::waitKey(1);
    if (key == 't' || key == 'T') timing_panel.toggle();
    if (key == 'r' || key == 'R') timing_panel.reset();
    if (key == 'q' || key == 'Q' || key == 27) {
      exit_reason = key == 27 ? "Esc key" : "q / Q key";
      break;
    }
    if (window_query_enabled) {
      try {
        // GTK can return -1 for WND_PROP_VISIBLE even while the window is open.
        // AUTOSIZE describes window existence without treating an unmapped/minimized
        // window as closed. Arm closure detection only after a successful query.
        const double autosize = cv::getWindowProperty(window, cv::WND_PROP_AUTOSIZE);
        if (std::isfinite(autosize) && autosize >= 0) {
          window_observed = true;
        } else if (window_observed) {
          window_closed = true;
          exit_reason = "GUI window closed";
          break;
        } else {
          window_query_enabled = false;
          tools::logger()->warn(
            "HighGUI 不支持窗口状态查询（返回 {}），继续运行；请按 q / Esc 或 Ctrl+C 退出", autosize);
        }
      } catch (const cv::Exception & e) {
        if (window_observed) {
          window_closed = true;
          exit_reason = "GUI window closed (window query unavailable after creation)";
          break;
        }
        window_query_enabled = false;
        tools::logger()->warn(
          "HighGUI 窗口状态查询不可用，继续运行；请按 q / Esc 或 Ctrl+C 退出：{}", e.what());
      }
    }
  }
  worker.stop();
  timing_panel.close();
  tools::logger()->info("普通自瞄可视化正常退出：{}，已处理 {} 帧", exit_reason, frame_count);
  if (!headless && !window_closed) cv::destroyWindow(window);
  return 0;
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>(0);
  const auto enable_control = cli.get<bool>("enable-control");
  const auto detect_timing = cli.get<bool>("detect-timing");
  const auto headless = cli.get<bool>("headless");
  const auto allow_unlit = cli.get<bool>("allow-unlit");
  const auto record = cli.get<bool>("record");
  const auto record_fps = cli.get<double>("record-fps");
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }
  if (!std::isfinite(record_fps) || record_fps <= 0) {
    tools::logger()->error("record-fps 必须为正数，当前为 {}", record_fps);
    return 1;
  }
  try {
    return run(config_path, enable_control, detect_timing, headless, allow_unlit, record, record_fps);
  } catch (const std::exception & e) {
    tools::logger()->error("普通自瞄可视化退出：{}", e.what());
    return 1;
  }
}