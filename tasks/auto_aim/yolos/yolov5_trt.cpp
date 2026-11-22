#include "yolov5_trt.hpp"

#ifdef SPVISION_WITH_TENSORRT

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
namespace
{
std::size_t data_type_size(nvinfer1::DataType type)
{
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      return 0;
  }
}

std::string dims_to_string(const nvinfer1::Dims & dims)
{
  std::string result = "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i) result += ", ";
    result += std::to_string(dims.d[i]);
  }
  return result + "]";
}

std::size_t volume(const nvinfer1::Dims & dims)
{
  std::size_t result = 1;
  for (int i = 0; i < dims.nbDims; ++i) result *= static_cast<std::size_t>(dims.d[i]);
  return result;
}

void check_cuda(cudaError_t status, const char * what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
      fmt::format("[trt] CUDA call '{}' failed: {}", what, cudaGetErrorString(status)));
  }
}

double event_ms(cudaEvent_t start, cudaEvent_t end)
{
  float ms = 0;
  check_cuda(cudaEventElapsedTime(&ms, start, end), "cudaEventElapsedTime");
  return ms;
}

std::vector<char> read_file(const std::string & path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("[trt] Cannot open engine file: " + path);
  const std::streamsize size = file.tellg();
  if (size <= 0) throw std::runtime_error("[trt] Engine file is empty: " + path);
  std::vector<char> data(static_cast<std::size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(data.data(), size))
    throw std::runtime_error("[trt] Failed to read engine file: " + path);
  return data;
}
}  // namespace

void YOLOV5TRTLogger::log(Severity severity, const char * msg) noexcept
{
  if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
    tools::logger()->error("[trt] {}", msg);
  } else if (severity == Severity::kWARNING) {
    tools::logger()->warn("[trt] {}", msg);
  }
}

YOLOV5TRT::YOLOV5TRT(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  if (!yaml["yolov5_trt_model_path"]) {
    throw std::runtime_error(
      "yolov5_trt_model_path is missing from " + config_path +
      " (expected e.g. assets/0708_fp16.engine, built on the Jetson by scripts/trt/build_engine.sh)");
  }
  model_path_ = yaml["yolov5_trt_model_path"].as<std::string>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  if (model_path_.size() > 7 && model_path_.compare(model_path_.size() - 7, 7, ".engine") != 0)
    tools::logger()->warn(
      "[trt] yolov5_trt_model_path \"{}\" does not end with .engine", model_path_);

  try {
    load_engine();
  } catch (...) {
    // A failed constructor does not run ~YOLOV5TRT; release any CUDA allocations.
    release();
    throw;
  }
  profiler_.context("YOLOV5TRT", fmt::format("TensorRT / {}", model_path_));

  tools::logger()->info(
    "[trt] Engine loaded: in {} {} {}, out {} {} {}", input_name_, dims_to_string(input_dims_),
    static_cast<int>(input_type_), output_name_, dims_to_string(output_dims_),
    static_cast<int>(output_type_));
}

YOLOV5TRT::~YOLOV5TRT() { release(); }

void YOLOV5TRT::release()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  for (auto & event : timing_events_) {
    if (!event) continue;
    cudaEventDestroy(event);
    event = nullptr;
  }
  if (input_host_) {
    cudaFreeHost(input_host_);
    input_host_ = nullptr;
  }
  if (output_host_) {
    cudaFreeHost(output_host_);
    output_host_ = nullptr;
  }
  if (input_device_) {
    cudaFree(input_device_);
    input_device_ = nullptr;
  }
  if (output_device_) {
    cudaFree(output_device_);
    output_device_ = nullptr;
  }
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

void YOLOV5TRT::load_engine()
{
  auto data = read_file(model_path_);

  runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
  if (!runtime_) throw std::runtime_error("[trt] createInferRuntime failed");

#ifdef SPVISION_TRT_V10
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
#else
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size(), nullptr));
#endif
  if (!engine_)
    throw std::runtime_error(
      "[trt] Cannot deserialize engine " + model_path_ +
      " (generated by a different TensorRT version or another GPU architecture?)");

  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("[trt] createExecutionContext failed");

  // Locate the single input and single output tensor.
#ifdef SPVISION_TRT_V10
  const int tensor_count = engine_->getNbIOTensors();
  for (int i = 0; i < tensor_count; ++i) {
    const char * name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      if (input_index_ >= 0) throw std::runtime_error("[trt] Engine has more than one input");
      input_index_ = i;
      input_name_ = name;
      input_dims_ = engine_->getTensorShape(name);
      input_type_ = engine_->getTensorDataType(name);
    } else {
      if (output_index_ >= 0) throw std::runtime_error("[trt] Engine has more than one output");
      output_index_ = i;
      output_name_ = name;
      output_dims_ = engine_->getTensorShape(name);
      output_type_ = engine_->getTensorDataType(name);
    }
  }
#else
  const int tensor_count = engine_->getNbBindings();
  for (int i = 0; i < tensor_count; ++i) {
    const char * name = engine_->getBindingName(i);
    if (engine_->bindingIsInput(i)) {
      if (input_index_ >= 0) throw std::runtime_error("[trt] Engine has more than one input");
      input_index_ = i;
      input_name_ = name;
      input_dims_ = engine_->getBindingDimensions(i);
      input_type_ = engine_->getBindingDataType(i);
    } else {
      if (output_index_ >= 0) throw std::runtime_error("[trt] Engine has more than one output");
      output_index_ = i;
      output_name_ = name;
      output_dims_ = engine_->getBindingDimensions(i);
      output_type_ = engine_->getBindingDataType(i);
    }
  }
#endif

  if (input_index_ < 0 || output_index_ < 0)
    throw std::runtime_error("[trt] Engine must expose exactly one input and one output");

  // The preprocessing below is hard-wired to 640x640 NCHW RGB; refuse anything else.
  const bool input_ok = input_dims_.nbDims == 4 && input_dims_.d[0] == 1 && input_dims_.d[1] == 3 &&
    input_dims_.d[2] == 640 && input_dims_.d[3] == 640;
  if (!input_ok)
    throw std::runtime_error(
      fmt::format(
        "[trt] Unexpected input shape {} (want [1, 3, 640, 640]); rebuild the engine from 0708.onnx",
        dims_to_string(input_dims_)));

  const std::size_t input_elem_size = data_type_size(input_type_);
  if (input_elem_size == 0 || (input_type_ != nvinfer1::DataType::kFLOAT &&
                               input_type_ != nvinfer1::DataType::kHALF))
    throw std::runtime_error(
      fmt::format("[trt] Unsupported input data type {}", static_cast<int>(input_type_)));

  if (output_dims_.nbDims != 3 || output_dims_.d[0] != 1 || output_dims_.d[1] <= 0 ||
      output_dims_.d[2] != 22)
    throw std::runtime_error(
      fmt::format(
        "[trt] Unexpected output shape {} (want [1, N, 22] with 4 corners + confidence + "
        "4 color + 9 number scores)",
        dims_to_string(output_dims_)));

  const std::size_t output_elem_size = data_type_size(output_type_);
  if (output_elem_size == 0 || (output_type_ != nvinfer1::DataType::kFLOAT &&
                                output_type_ != nvinfer1::DataType::kHALF))
    throw std::runtime_error(
      fmt::format("[trt] Unsupported output data type {}", static_cast<int>(output_type_)));

  output_rows_ = static_cast<int>(output_dims_.d[1]);
  output_cols_ = static_cast<int>(output_dims_.d[2]);
  input_bytes_ = volume(input_dims_) * input_elem_size;
  output_bytes_ = volume(output_dims_) * output_elem_size;

  check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  check_cuda(cudaMallocHost(&input_host_, input_bytes_), "cudaMallocHost(input)");
  check_cuda(cudaMallocHost(&output_host_, output_bytes_), "cudaMallocHost(output)");
  check_cuda(cudaMalloc(&input_device_, input_bytes_), "cudaMalloc(input)");
  check_cuda(cudaMalloc(&output_device_, output_bytes_), "cudaMalloc(output)");

#ifdef SPVISION_TRT_ENQUEUE_V3
  if (!context_->setTensorAddress(input_name_.c_str(), input_device_))
    throw std::runtime_error("[trt] setTensorAddress(input) failed");
  if (!context_->setTensorAddress(output_name_.c_str(), output_device_))
    throw std::runtime_error("[trt] setTensorAddress(output) failed");
#else
  bindings_.assign(static_cast<std::size_t>(tensor_count), nullptr);
  bindings_[static_cast<std::size_t>(input_index_)] = input_device_;
  bindings_[static_cast<std::size_t>(output_index_)] = output_device_;
#endif
}

void YOLOV5TRT::enqueue()
{
#ifdef SPVISION_TRT_ENQUEUE_V3
  if (!context_->enqueueV3(stream_)) throw std::runtime_error("[trt] enqueueV3 failed");
#else
  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr))
    throw std::runtime_error("[trt] enqueueV2 failed");
#endif
}

void YOLOV5TRT::pack_input(const cv::Mat & bgr_canvas)
{
  // Same normalisation as the OpenCV DNN path: BGR u8 -> RGB f32 / 255, NCHW.
  cv::Mat rgb;
  cv::cvtColor(bgr_canvas, rgb, cv::COLOR_BGR2RGB);
  cv::Mat rgb_f;
  rgb.convertTo(rgb_f, CV_32F, 1.0 / 255.0);

  std::vector<cv::Mat> planes;
  cv::split(rgb_f, planes);

  const std::size_t elem_size = data_type_size(input_type_);
  const std::size_t plane_bytes = planes[0].total() * elem_size;
  auto * dst = static_cast<std::uint8_t *>(input_host_);
  for (int c = 0; c < 3; ++c) {
    if (input_type_ == nvinfer1::DataType::kFLOAT) {
      std::memcpy(dst + c * plane_bytes, planes[c].data, plane_bytes);
    } else {
      cv::Mat half;
      planes[c].convertTo(half, CV_16F);
      std::memcpy(dst + c * plane_bytes, half.data, plane_bytes);
    }
  }
}

cv::Mat YOLOV5TRT::fetch_output()
{
  cv::Mat result;
  if (output_type_ == nvinfer1::DataType::kFLOAT) {
    result = cv::Mat(output_rows_, output_cols_, CV_32F, output_host_).clone();
  } else {
    // FP16 engines may keep the output in half precision; never assume float32.
    cv::Mat half(output_rows_, output_cols_, CV_16F, output_host_);
    half.convertTo(result, CV_32F);
  }
  return result;
}

std::vector<InferenceGraph> YOLOV5TRT::inference_graphs()
{
  return {{"detector", "",
    "TensorRT engine " + model_path_ +
      "; the serialized engine has no inspectable graph. Number classes come from the "
      "detection head; optional traditional corner refinement runs on CPU."}};
}

std::list<Armor> YOLOV5TRT::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  profiler_.mark(DetectStage::roi);
  // 与 OpenVINO / OpenCV DNN 原实现一致：左上角放置，黑色填充
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});
  profiler_.mark(DetectStage::resize);

  pack_input(input);
  profiler_.mark(DetectStage::pack);

  const bool timed = profiler_.active();
  const auto & [upload_start, infer_start, download_start, download_end] = timing_events_;
  if (timed) {
    for (auto & event : timing_events_) {
      if (!event) check_cuda(cudaEventCreate(&event), "cudaEventCreate");
    }
  }

  try {
    if (timed) check_cuda(cudaEventRecord(upload_start, stream_), "cudaEventRecord(upload start)");
    check_cuda(
      cudaMemcpyAsync(input_device_, input_host_, input_bytes_, cudaMemcpyHostToDevice, stream_),
      "cudaMemcpyAsync(input)");
    if (timed) check_cuda(cudaEventRecord(infer_start, stream_), "cudaEventRecord(infer start)");

    enqueue();

    if (timed)
      check_cuda(cudaEventRecord(download_start, stream_), "cudaEventRecord(download start)");
    check_cuda(
      cudaMemcpyAsync(output_host_, output_device_, output_bytes_, cudaMemcpyDeviceToHost, stream_),
      "cudaMemcpyAsync(output)");
    if (timed) check_cuda(cudaEventRecord(download_end, stream_), "cudaEventRecord(download end)");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  } catch (...) {
    // Drain queued work before a later detect() can reuse the pinned buffers.
    cudaStreamSynchronize(stream_);
    throw;
  }

  if (timed) {
    profiler_.mark_device({
      {DetectStage::bind, event_ms(upload_start, infer_start)},
      {DetectStage::forward, event_ms(infer_start, download_start)},
      {DetectStage::output, event_ms(download_start, download_end)},
    });
  }

  auto output = fetch_output();
  profiler_.mark(DetectStage::output);

  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> YOLOV5TRT::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: 8 corner values + confidence + color scores + number scores
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  for (int r = 0; r < output.rows; r++) {
    double score = output.at<float>(r, 8);
    score = sigmoid(score);

    if (score < score_threshold_) continue;

    std::vector<cv::Point2f> armor_key_points;

    //颜色和类别独热向量
    cv::Mat color_scores = output.row(r).colRange(9, 13);     //color
    cv::Mat classes_scores = output.row(r).colRange(13, 22);  //num
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    _class_id = class_id.x;
    _color_id = color_id.x;

    // 原始顺序 TL, BL, BR, TR -> Armor.points 顺序 TL, TR, BR, BL
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 0) / scale, output.at<float>(r, 1) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 6) / scale, output.at<float>(r, 7) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 4) / scale, output.at<float>(r, 5) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 2) / scale, output.at<float>(r, 3) / scale));

    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (int i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  profiler_.mark(DetectStage::decode);
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);
  profiler_.mark(DetectStage::nms);
  profiler_.candidates(output.rows, static_cast<int>(boxes.size()), static_cast<int>(indices.size()));

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  tmp_img_ = bgr_img;
  profiler_.mark(DetectStage::construct);
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      profiler_.mark(DetectStage::filter);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      profiler_.mark(DetectStage::filter);
      continue;
    }
    profiler_.mark(DetectStage::filter);
    // 使用传统方法二次矫正角点
    if (use_traditional_) {
      detector_.detect(*it, bgr_img);
      profiler_.mark(DetectStage::refine);
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
    profiler_.mark(DetectStage::filter);
  }

  profiler_.mark(DetectStage::filter);
  if (debug_) {
    draw_detections(bgr_img, armors, frame_count);
    profiler_.mark(DetectStage::debug_draw);
  }

  return armors;
}

bool YOLOV5TRT::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  return name_ok && confidence_ok;
}

bool YOLOV5TRT::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  return name_ok;
}

cv::Point2f YOLOV5TRT::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5TRT::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLOV5TRT::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5TRT::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5TRT::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim

#endif  // SPVISION_WITH_TENSORRT
