#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

namespace auto_aim
{
enum class DetectStage : std::size_t
{
  roi, resize, pack, request, bind, forward, output, layout, decode, nms,
  construct, pattern, classify, refine, filter, debug_draw, other, count
};

inline constexpr std::size_t detect_stage_count = static_cast<std::size_t>(DetectStage::count);
inline constexpr std::array<const char *, detect_stage_count> detect_stage_names = {
  "ROI / shape setup", "Resize / padding", "Blob / tensor packing", "Create infer request",
  "Input binding (setInput)", "Network forward / infer", "Output view / copy", "Output transpose",
  "Decode / score filtering", "NMS", "Armor / keypoint construction", "Classifier image crop",
  "Number classification", "Traditional corner refinement", "Armor filtering / normalization",
  "Detector debug drawing", "Other / dispatch"};

struct DetectionTiming
{
  bool valid = false;
  bool postprocess_only = false;
  bool device_timing = false;
  uint64_t frame = 0;
  std::string model, backend;
  double total_ms = 0;
  std::array<double, detect_stage_count> stage_ms{};
  std::array<bool, detect_stage_count> measured{};
  int raw_candidates = -1, score_candidates = -1, nms_candidates = -1, armors = -1;

  double ms(DetectStage stage) const { return stage_ms[static_cast<std::size_t>(stage)]; }
  double preprocessing_ms() const
  {
    return ms(DetectStage::roi) + ms(DetectStage::resize) + ms(DetectStage::pack) +
      ms(DetectStage::request) + ms(DetectStage::bind);
  }
  double postprocessing_ms() const
  {
    double result = 0;
    for (std::size_t i = static_cast<std::size_t>(DetectStage::output);
         i < static_cast<std::size_t>(DetectStage::other); ++i)
      result += stage_ms[i];
    return result;
  }
};

// Disabled by default. No clock queries or per-candidate timers on the disabled path.
// Each checkpoint accounts for a wall-clock interval that can include measured device stages.
class DetectionProfiler
{
public:
  void enable(bool value) { enabled_ = value; active_ = false; timing_.valid = false; }
  bool active() const { return active_; }
  const DetectionTiming & timing() const { return timing_; }

  void context(std::string model, std::string backend)
  {
    timing_.model = std::move(model);
    timing_.backend = std::move(backend);
  }

  void begin(bool postprocess_only = false)
  {
    timing_.valid = false;
    active_ = enabled_;
    if (!active_) return;
    timing_.postprocess_only = postprocess_only;
    timing_.device_timing = false;
    timing_.stage_ms.fill(0);
    timing_.measured.fill(false);
    timing_.total_ms = 0;
    timing_.raw_candidates = timing_.score_candidates = timing_.nms_candidates = timing_.armors = -1;
    ++timing_.frame;
    start_ = checkpoint_ = Clock::now();
  }

  void mark(DetectStage stage)
  {
    if (!active_) return;
    const auto now = Clock::now();
    add(stage, std::chrono::duration<double, std::milli>(now - checkpoint_).count());
    checkpoint_ = now;
  }

  void mark_device(std::initializer_list<std::pair<DetectStage, double>> stages)
  {
    if (!active_) return;
    const auto now = Clock::now();
    double remaining_ms = std::chrono::duration<double, std::milli>(now - checkpoint_).count();
    for (const auto & [stage, ms] : stages) {
      add(stage, ms);
      remaining_ms -= ms;
    }
    add(DetectStage::other, std::max(0.0, remaining_ms));
    checkpoint_ = now;
    timing_.device_timing = true;
  }

  void candidates(int raw, int scored, int nms)
  {
    if (!active_) return;
    timing_.raw_candidates = raw;
    timing_.score_candidates = scored;
    timing_.nms_candidates = nms;
  }

  void finish(std::size_t armors)
  {
    if (!active_) return;
    const auto now = Clock::now();
    add(DetectStage::other, std::chrono::duration<double, std::milli>(now - checkpoint_).count());
    timing_.total_ms = std::chrono::duration<double, std::milli>(now - start_).count();
    timing_.armors = static_cast<int>(armors);
    timing_.valid = true;
    active_ = false;
  }

private:
  void add(DetectStage stage, double ms)
  {
    const auto index = static_cast<std::size_t>(stage);
    timing_.stage_ms[index] += ms;
    timing_.measured[index] = true;
  }

  using Clock = std::chrono::steady_clock;
  bool enabled_ = false, active_ = false;
  DetectionTiming timing_;
  Clock::time_point start_, checkpoint_;
};
}  // namespace auto_aim
