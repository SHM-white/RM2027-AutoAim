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
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "detection_timing_panel.hpp"

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | yaml 配置文件路径 }"
  "{enable-control | true                | 开启云台跟随，默认开启，始终不开火 }"
  "{detect-timing  | true                | 显示 detect 分段耗时及最近 60 帧统计 }";

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

struct PlanView
{
  auto_aim::Plan plan{};
  std::optional<auto_aim::PlannerTrace> trace;
  Clock::time_point source_time{};
  Clock::time_point computed{};
  double solve_ms = 0;
  bool tx_fire_request = false;
  std::string status = "Waiting for live image and pose";
  std::string control_status = "RX ONLY - no serial TX";
};

// Owns the planner and its trace. The GUI receives copies under one mutex.
// The optional gimbal is the only command writer. A null pointer preserves RX-only mode.
class PlanningWorker
{
public:
  PlanningWorker(const std::string & config_path, io::Gimbal * control_gimbal)
  : planner_(config_path, true), control_gimbal_(control_gimbal), thread_([this] { run(); })
  {}

  ~PlanningWorker() { stop(); }

  void stop()
  {
    quit_ = true;
    if (thread_.joinable()) thread_.join();
  }

  void submit(const std::optional<auto_aim::Target> & target, Clock::time_point t,
              bool pose_valid, bool tracking_ready, double bullet_speed, uint64_t generation)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool same_target = input_.target && target &&
      input_.target->name == target->name && input_.target->armor_type == target->armor_type;
    const bool ready = pose_valid && tracking_ready && target.has_value();
    if (!ready || !input_.pose_valid || !input_.tracking_ready || !same_target ||
        input_.generation != generation)
      ++input_.epoch;
    input_ = {target, t, pose_valid, tracking_ready, bullet_speed, generation, input_.epoch};
  }

  PlanView view()
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
    uint64_t epoch = 0;
  };

  void transmit(const Input & input, PlanView & output)
  {
    if (!control_gimbal_) return;
    const auto now = Clock::now();
    const auto feedback = control_gimbal_->feedback();
    bool current_input;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_input = input.epoch == input_.epoch;
    }
    const std::array<double, 6> command = {
      output.plan.yaw, output.plan.yaw_vel, output.plan.yaw_acc,
      output.plan.pitch, output.plan.pitch_vel, output.plan.pitch_acc};
    auto_aim::Plan tx{};
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
    else if (!output.plan.control || !output.trace ||
             !std::all_of(command.begin(), command.end(), [](double x) { return std::isfinite(x); }))
      output.control_status = "TX STOP: no valid MPC command";
    else {
      tx = output.plan;
      output.control_status = "TX FOLLOW";
    }
    output.tx_fire_request = tx.control && tx.fire;
    control_gimbal_->send(
      tx.control, tx.fire, tx.yaw, tx.yaw_vel, tx.yaw_acc, tx.pitch, tx.pitch_vel, tx.pitch_acc);
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
      PlanView output;
      output.source_time = input.t;
      if (!input.pose_valid)
        output.status = "No image-matched pose: planning paused";
      else if (start - input.t > 250ms)
        output.status = "Image / target older than 250 ms: planning paused";
      else if (!input.target)
        output.status = "No tracked target";
      else {
        try {
          output.plan = planner_.plan(input.target, input.bullet_speed);
          if (output.plan.control && planner_.trace() &&
              planner_.trace()->reference.allFinite() &&
              planner_.trace()->solution.allFinite() &&
              std::isfinite(output.plan.yaw) && std::isfinite(output.plan.pitch)) {
            output.trace = planner_.trace();
            output.status = "Live MPC result";
          } else {
            output.plan = {};
            output.status = "No valid ballistic / MPC solution";
          }
        } catch (const std::exception & e) {
          output.plan = {};
          output.status = std::string("Planner: ") + e.what();
        }
      }
      output.computed = Clock::now();
      output.solve_ms = std::chrono::duration<double, std::milli>(output.computed - start).count();
      transmit(input, output);
      if (control_gimbal_ && output.control_status != last_control_status) {
        tools::logger()->info("[AutoAim GUI] {}", output.control_status);
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
      tools::logger()->info("[AutoAim GUI] 控制线程退出，已请求停控、不开火");
    }
  }

  auto_aim::Planner planner_;
  io::Gimbal * control_gimbal_;
  std::mutex mutex_;
  Input input_;
  PlanView output_;
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
  const Projection & projection, const PlanView & view, bool pose_valid)
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
  if (!view.trace || !view.plan.control) return;
  const auto & trace = *view.trace;
  std::vector<Eigen::Vector3d> aim_path, command_path;
  const double distance = std::max(0.2, trace.aim_xyza[auto_aim::HALF_HORIZON].head<3>().norm());
  for (int i = auto_aim::HALF_HORIZON; i < auto_aim::HORIZON; ++i) {
    aim_path.push_back(trace.aim_xyza[i].head<3>());
    command_path.push_back(
      from_ypr(trace.solution(0, i), trace.solution(2, i)) * Eigen::Vector3d::UnitX() * distance);
  }
  draw_path(image, projection.pixels(aim_path, rotation, display_scale), reference_color);
  draw_path(image, projection.pixels(command_path, rotation, display_scale), command_color, 3);
  const auto aim = projection.pixels(
    {trace.aim_xyza[auto_aim::HALF_HORIZON].head<3>()}, rotation, display_scale);
  if (aim.front() && cv::Rect(0, 0, image.cols, image.rows).contains(*aim.front()))
    cv::drawMarker(image, *aim.front(), reference_color, cv::MARKER_DIAMOND, 22, 2);
  const Eigen::Vector3d current = rotation * Eigen::Vector3d::UnitX() * distance;
  const Eigen::Vector3d command =
    from_ypr(view.plan.yaw, view.plan.pitch) * Eigen::Vector3d::UnitX() * distance;
  const auto direction = projection.pixels({current, command}, rotation, display_scale);
  if (direction[0] && direction[1]) {
    auto a = *direction[0], b = *direction[1];
    if (cv::clipLine(image.size(), a, b))
      cv::arrowedLine(image, a, b, command_color, 3, cv::LINE_AA, 0, 0.15);
  }
  if (direction[1] && cv::Rect(0, 0, image.cols, image.rows).contains(*direction[1]))
    cv::drawMarker(image, *direction[1], command_color, cv::MARKER_CROSS, 26, 2);
}

void draw_curve(cv::Mat & canvas, const cv::Rect & area, const PlanView & view, int row,
                const std::string & title, std::optional<double> received)
{
  cv::rectangle(canvas, area, cv::Scalar(65, 65, 75));
  text(canvas, title + " (deg)", area.x + 10, area.y + 20);
  if (!view.trace) {
    text(canvas, "Waiting for a live MPC solution", area.x + 16, area.y + 85, muted);
    return;
  }
  const auto & trace = *view.trace;
  double low = std::min(trace.reference.row(row).minCoeff(), trace.solution.row(row).minCoeff());
  double high = std::max(trace.reference.row(row).maxCoeff(), trace.solution.row(row).maxCoeff());
  if (received) {
    if (row == 0) {
      const double center = trace.reference(0, auto_aim::HALF_HORIZON);
      *received = center + std::remainder(*received - center, 2 * CV_PI);
    }
    low = std::min(low, *received);
    high = std::max(high, *received);
  }
  const double pad = std::max((high - low) * 0.12, 0.25 / degrees);
  low -= pad;
  high += pad;
  const cv::Rect plot(area.x + 60, area.y + 35, area.width - 74, area.height - 62);
  const auto point = [&](int i, double value) {
    return cv::Point(plot.x + cvRound(i * (plot.width - 1.0) / (auto_aim::HORIZON - 1)),
      plot.y + cvRound((high - value) / (high - low) * (plot.height - 1)));
  };
  for (int i = 0; i <= 2; ++i) {
    const double value = low + (high - low) * i / 2;
    const auto p = point(0, value);
    cv::line(canvas, p, {plot.x + plot.width, p.y}, {55, 55, 65});
    text(canvas, fmt::format("{:.1f}", value * degrees), area.x + 4, p.y + 4, muted, 0.4);
  }
  for (int i = 1; i < auto_aim::HORIZON; ++i) {
    cv::line(canvas, point(i - 1, trace.reference(row, i - 1)),
      point(i, trace.reference(row, i)), reference_color, 1, cv::LINE_AA);
    cv::line(canvas, point(i - 1, trace.solution(row, i - 1)),
      point(i, trace.solution(row, i)), command_color, 2, cv::LINE_AA);
  }
  if (received)
    cv::line(canvas, point(0, *received), point(auto_aim::HORIZON - 1, *received),
      feedback_color, 1, cv::LINE_AA);
  const auto now = point(auto_aim::HALF_HORIZON, low);
  cv::line(canvas, {now.x, plot.y}, {now.x, plot.y + plot.height}, muted, 1);
  text(canvas, "-0.50 s", plot.x, area.y + area.height - 7, muted, 0.4);
  text(canvas, "0", now.x - 4, area.y + area.height - 7, muted, 0.4);
  text(canvas, "+0.49 s", plot.x + plot.width - 58, area.y + area.height - 7, muted, 0.4);
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

int run(const std::string & config_path, bool enable_control, bool detect_timing)
{
  const auto yaml = YAML::LoadFile(config_path);
  const double default_speed = yaml["default_bullet_speed"].as<double>();
  if (!std::isfinite(default_speed) || default_speed < 10 || default_speed > 25)
    throw std::runtime_error("default_bullet_speed must be within 10..25 m/s for this MPC planner");
  const Projection projection(yaml);
  tools::Exiter exiter;
  // Only the opt-in planning worker can send; no IMU forwarding or navigation callbacks.
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path, enable_control
    ? io::CameraFramePolicy::keep_queued : io::CameraFramePolicy::keep_latest);
  auto_aim::YOLO yolo(config_path, false);
  yolo.set_timing_enabled(detect_timing);
  DetectionTimingPanel timing_panel(detect_timing);
  auto_aim::Solver solver(config_path);
  auto tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
  PlanningWorker worker(config_path, enable_control ? &gimbal : nullptr);
  tools::logger()->info("自瞄实机可视化：{}", enable_control
    ? "已启用云台跟随，开火位始终关闭" : "仅接收串口反馈，不发送任何控制/开火帧");
  tools::logger()->info("弹速反馈不可用时读取配置默认值 {} m/s", default_speed);
  if (detect_timing)
    tools::logger()->info("detect 分段计时已开启：T 显示/隐藏计时窗口，R 清除最近 60 帧统计");

  const std::string window = enable_control ? "Auto aim MPC - FOLLOW / NO FIRE" : "Auto aim MPC - RX ONLY";
  cv::namedWindow(window, cv::WINDOW_NORMAL);
  cv::resizeWindow(window, 1280, 800);
  const cv::Size canvas_size(1600, 1000);
  const cv::Scalar canvas_background(24, 25, 30);
  cv::Mat preview_canvas;
  if (!enable_control) preview_canvas.create(canvas_size, CV_8UC3);
  cv::Mat image;
  Clock::time_point image_time;
  int frame_count = 0;
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
    }
    had_pose = pose_valid;
    const double board_speed = feedback ? feedback->state.bullet_speed : 0;
    const bool use_default = !std::isfinite(board_speed) || board_speed < 10 || board_speed > 25;
    const double speed = use_default ? default_speed : board_speed;
    worker.submit(target, image_time, pose_valid, tracker->state() == "tracking", speed,
      feedback ? feedback->generation : 0);
    const auto tracking_finish = Clock::now();
    auto view = worker.view();
    if (!pose_valid || !target || view.source_time == Clock::time_point{} ||
        Clock::now() - view.source_time > 250ms) {
      view.plan = {};
      view.trace.reset();
      view.status = !pose_valid ? "No image-matched pose: planning paused" :
        (!target ? "No tracked target" : "Waiting for a fresh MPC result");
    }
    std::optional<Eigen::Quaterniond> raw_q, raw_direction, command_q;
    if (feedback) {
      raw_q = feedback->q;
      if (eulers(*raw_q)) raw_direction = projection.raw_in_gimbal_world(*raw_q);
    }
    if (view.plan.control && view.trace)
      command_q = from_ypr(view.plan.yaw, view.plan.pitch);

    cv::Mat canvas = enable_control ? cv::Mat(canvas_size, CV_8UC3) : preview_canvas;
    canvas.setTo(canvas_background);
    text(canvas, fmt::format("AUTO AIM: CAMERA -> DETECTION -> TRACKER -> MPC | {} | q / Esc exit",
      enable_control ? "FOLLOW / FIRE OFF" : "RX ONLY"),
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
    text(canvas, "Blue detection | Green track | Yellow aim prediction | Magenta MPC / command", 18, 795,
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
    draw_pose(canvas, 282, "DESIRED GIMBAL q (MPC command; roll = 0)", command_q, command_color);
    if (command_q && feedback) {
      text(canvas, fmt::format("CMD - packet Y/P: {:+.2f} / {:+.2f} deg",
        std::remainder(view.plan.yaw - feedback->state.yaw, 2 * CV_PI) * degrees,
        std::remainder(view.plan.pitch - feedback->state.pitch, 2 * CV_PI) * degrees), 990, 354);
      text(canvas, fmt::format("CMD vel Y/P {:+.2f}/{:+.2f} deg/s | planned fire {}",
        view.plan.yaw_vel * degrees, view.plan.pitch_vel * degrees, view.plan.fire ? "YES" : "NO"),
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
    if (view.trace)
      text(canvas, fmt::format("Fly time {:.3f} s | source age {:.1f} ms | MPC {:.1f} ms",
        view.trace->fly_time, std::chrono::duration<double, std::milli>(now - view.source_time).count(),
        view.solve_ms),
        990, 600, muted, 0.49);
    text(canvas, view.status.substr(0, 68), 990, 630, command_q ? tracked_color : reference_color, 0.49);
    text(canvas, fmt::format("Configured offsets Y/P {:+.2f} / {:+.2f} deg",
      yaml["yaw_offset"].as<double>(), yaml["pitch_offset"].as<double>()), 990, 654, muted, 0.49);
    const auto raw_angles = raw_direction ? eulers(*raw_direction) : std::nullopt;
    draw_curve(canvas, {10, 815, 475, 180}, view, 0, "Yaw: REF / MPC / RX",
      raw_angles ? std::optional<double>((*raw_angles)[0]) : std::nullopt);
    draw_curve(canvas, {495, 815, 475, 180}, view, 2, "Pitch: REF / MPC / RX",
      raw_angles ? std::optional<double>((*raw_angles)[1]) : std::nullopt);
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
  tools::logger()->info("自瞄可视化正常退出：{}，已处理 {} 帧", exit_reason, frame_count);
  if (!window_closed) cv::destroyWindow(window);
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
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }
  try {
    return run(config_path, enable_control, detect_timing);
  } catch (const std::exception & e) {
    tools::logger()->error("自瞄可视化退出：{}", e.what());
    return 1;
  }
}
