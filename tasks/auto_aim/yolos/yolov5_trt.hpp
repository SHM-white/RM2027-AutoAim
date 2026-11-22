#ifndef AUTO_AIM__YOLOV5_TRT_HPP
#define AUTO_AIM__YOLOV5_TRT_HPP

#include <array>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// CMake includes TensorRT only when it selects the installed TensorRT runtime.
// Otherwise this header stays empty and does not require CUDA headers.
#ifdef SPVISION_WITH_TENSORRT

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

// TensorRT 10 removed the deprecated binding API and destroy(); 8.x still needs them.
#if NV_TENSORRT_MAJOR >= 10
#define SPVISION_TRT_V10 1
#endif

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5)
#define SPVISION_TRT_ENQUEUE_V3 1
#endif

namespace auto_aim
{
// TensorRT 8.x interfaces have a protected destructor (destroy() required);
// TensorRT 10 removed destroy() and requires delete.
struct TrtDeleter
{
  template <typename T>
  void operator()(T * obj) const
  {
    if (!obj) return;
#ifdef SPVISION_TRT_V10
    delete obj;
#else
    obj->destroy();
#endif
  }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter>;

// Routes TensorRT deserialization messages into this project's logger.
class YOLOV5TRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

// TensorRT backend for the Shenzhen University 0708.onnx detector
// (input [1,3,640,640] RGB/255, output [1,25200,22]). Pre- and post-processing
// replicate YOLOV5 exactly so Solver/Tracker/Aimer see identical Armor data.
class YOLOV5TRT : public YOLOBase
{
public:
  YOLOV5TRT(const std::string & config_path, bool debug);
  ~YOLOV5TRT() override;

  YOLOV5TRT(const YOLOV5TRT &) = delete;
  YOLOV5TRT & operator=(const YOLOV5TRT &) = delete;

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;
  std::vector<InferenceGraph> inference_graphs() override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string model_path_;
  std::string save_path_;
  bool debug_, use_roi_, use_traditional_;

  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;

  YOLOV5TRTLogger trt_logger_;
  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  TrtPtr<nvinfer1::IExecutionContext> context_;

  std::string input_name_, output_name_;
  nvinfer1::DataType input_type_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType output_type_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::Dims input_dims_{}, output_dims_{};
  std::size_t input_bytes_ = 0, output_bytes_ = 0;
  int input_index_ = -1, output_index_ = -1;
  int output_rows_ = 0, output_cols_ = 0;

  cudaStream_t stream_ = nullptr;
  std::array<cudaEvent_t, 4> timing_events_{};
  void * input_host_ = nullptr;
  void * output_host_ = nullptr;
  void * input_device_ = nullptr;
  void * output_device_ = nullptr;
  std::vector<void *> bindings_;

  void load_engine();
  void pack_input(const cv::Mat & bgr_canvas);
  void enqueue();
  cv::Mat fetch_output();
  void release();

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);
};

}  // namespace auto_aim

#endif  // SPVISION_WITH_TENSORRT

#endif  // AUTO_AIM__YOLOV5_TRT_HPP
