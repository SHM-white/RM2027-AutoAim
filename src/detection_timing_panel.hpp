#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fmt/core.h>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/detection_timing.hpp"

// The panel is updated on the GUI thread, outside the measured detect() interval.
class DetectionTimingPanel
{
public:
  explicit DetectionTimingPanel(bool enabled) : enabled_(enabled), visible_(enabled) {}
  ~DetectionTimingPanel() { close(); }

  void add(const auto_aim::DetectionTiming & timing)
  {
    if (!enabled_ || !timing.valid) return;
    history_.push_back(timing);
    if (history_.size() > 60) history_.pop_front();
  }

  void reset()
  {
    history_.clear();
    last_draw_ = {};
  }

  void toggle()
  {
    if (!enabled_) return;
    if (visible_) close();
    else { visible_ = true; last_draw_ = {}; }
  }

  void close() noexcept
  {
    visible_ = false;
    if (!created_) return;
    try { cv::destroyWindow(window_); }
    catch (const cv::Exception &) {}
    created_ = false;
  }

  void show()
  {
    if (!enabled_ || !visible_ || history_.empty()) return;
    if (created_ && query_enabled_) {
      try {
        const double property = cv::getWindowProperty(window_, cv::WND_PROP_AUTOSIZE);
        if (std::isfinite(property) && property >= 0) window_observed_ = true;
        else if (window_observed_) { created_ = false; visible_ = false; return; }
        else query_enabled_ = false;
      } catch (const cv::Exception &) {
        if (window_observed_) { created_ = false; visible_ = false; return; }
        query_enabled_ = false;
      }
    }
    if (!created_) {
      cv::namedWindow(window_, cv::WINDOW_NORMAL);
      cv::resizeWindow(window_, 1120, 760);
      created_ = true;
      window_observed_ = false;
      query_enabled_ = true;
      last_draw_ = {};
    }
    const auto now = Clock::now();
    if (now - last_draw_ < std::chrono::milliseconds(200)) return;
    cv::imshow(window_, render());
    last_draw_ = now;
  }

private:
  using Clock = std::chrono::steady_clock;
  struct Statistics
  {
    double mean = 0, p95 = 0, maximum = 0;
    bool valid = false;
  };

  static Statistics summarize(std::vector<double> values)
  {
    if (values.empty()) return {};
    Statistics result;
    result.valid = true;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    result.p95 = values[static_cast<std::size_t>(std::ceil(0.95 * values.size())) - 1];
    result.maximum = values.back();
    return result;
  }

  static void text(cv::Mat & image, const std::string & value, int x, int y,
                   cv::Scalar color = {225, 228, 235}, double scale = 0.49)
  {
    cv::putText(image, value, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
  }

  static std::string number(double value, bool valid)
  {
    return valid ? fmt::format("{:8.3f}", value) : "      --";
  }

  cv::Mat render() const
  {
    using auto_aim::DetectStage;
    const auto & current = history_.back();
    cv::Mat image(760, 1120, CV_8UC3, cv::Scalar(25, 26, 33));
    const cv::Scalar accent(70, 220, 250), muted(145, 150, 165);
    text(image, "DETECT TIMING (ms) | T show/hide | R reset statistics", 18, 29, accent, 0.65);
    text(image, current.model + " | " + current.backend, 18, 56, {225, 228, 235}, 0.55);
    text(image, fmt::format("Frame {} | raw {} -> score {} -> NMS {} -> armors {} | rolling {} frames",
      current.frame, current.raw_candidates, current.score_candidates, current.nms_candidates,
      current.armors, history_.size()), 18, 83);
    text(image, current.device_timing
      ? "TensorRT transfers and inference use CUDA events. Total uses wall-clock time."
      : "Stages use wall-clock time. The backend label shows the requested device.",
      18, 109, muted, 0.45);
    text(image, "Stage", 18, 143, muted);
    text(image, "Last", 352, 143, muted);
    text(image, "Mean", 463, 143, muted);
    text(image, "P95", 580, 143, muted);
    text(image, "Max", 694, 143, muted);
    text(image, "% total", 795, 143, muted);
    text(image, "Current share", 910, 143, muted);

    std::array<Statistics, auto_aim::detect_stage_count> stages;
    std::vector<double> totals;
    for (const auto & frame : history_) totals.push_back(frame.total_ms);
    const auto total = summarize(totals);
    for (std::size_t i = 0; i < auto_aim::detect_stage_count; ++i) {
      std::vector<double> values;
      for (const auto & frame : history_)
        if (frame.measured[i]) values.push_back(frame.stage_ms[i]);
      stages[i] = summarize(values);
      const int y = 170 + static_cast<int>(i) * 25;
      const bool executed = current.measured[i];
      const auto color = i == static_cast<std::size_t>(DetectStage::forward) ? accent : cv::Scalar(225, 228, 235);
      const auto name = current.device_timing && i == static_cast<std::size_t>(DetectStage::bind)
        ? "Input upload (H2D)" : auto_aim::detect_stage_names[i];
      text(image, name, 18, y, executed ? color : muted, 0.45);
      text(image, number(current.stage_ms[i], executed), 316, y, color);
      text(image, number(stages[i].mean, stages[i].valid), 430, y);
      text(image, number(stages[i].p95, stages[i].valid), 544, y);
      text(image, number(stages[i].maximum, stages[i].valid), 658, y);
      const double share = current.total_ms > 0 ? current.stage_ms[i] / current.total_ms : 0;
      text(image, executed ? fmt::format("{:5.1f}%", share * 100) : "    --", 790, y, muted);
      cv::rectangle(image, {902, y - 13, 196, 14}, cv::Scalar(50, 52, 65), cv::FILLED);
      const int width = cvRound(std::clamp(share, 0.0, 1.0) * 196);
      if (width > 0) cv::rectangle(image, {902, y - 13, width, 14}, color, cv::FILLED);
    }
    cv::line(image, {18, 590}, {1098, 590}, muted);
    text(image, "DETECT TOTAL", 18, 620, accent, 0.58);
    text(image, number(current.total_ms, true), 316, 620, accent, 0.58);
    text(image, number(total.mean, true), 430, 620, accent, 0.58);
    text(image, number(total.p95, true), 544, 620, accent, 0.58);
    text(image, number(total.maximum, true), 658, 620, accent, 0.58);
    text(image, fmt::format("Pre {:.3f} | forward {:.3f} | post {:.3f} | other {:.3f} ms",
      current.preprocessing_ms(), current.ms(DetectStage::forward), current.postprocessing_ms(),
      current.ms(DetectStage::other)), 18, 654, accent, 0.53);
    text(image, current.device_timing
      ? "Output time includes the device-to-host copy and host output preparation."
      : "Forward time may include transfers, synchronization, conversions and CPU fallback.",
      18, 682, muted, 0.43);
    text(image, "-- means this stage was not executed. Stage statistics use only frames that executed that stage.",
      18, 708, muted, 0.43);
    text(image, current.postprocess_only
      ? "Postprocess-only call: preprocessing / inference are outside this measurement."
      : "Detector timing excludes camera capture, tracking, planning and this GUI. Startup frames remain in the window.",
      18, 734, muted, 0.43);
    return image;
  }

  bool enabled_ = false, visible_ = false, created_ = false;
  bool query_enabled_ = true, window_observed_ = false;
  const std::string window_ = "Detect timing (ms)";
  std::deque<auto_aim::DetectionTiming> history_;
  Clock::time_point last_draw_{};
};
