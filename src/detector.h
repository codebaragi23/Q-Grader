#pragma once

#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"

#include <opencv2/opencv.hpp>

typedef struct
{
  cv::Rect  bbox;
  int   labelid;
  float score;
} BoxInfo;

class QGDetector
{
public:
  typedef struct Params
  {
    int width = 640;
    int height = 640;
    int channel = 3;

    int num_classes = 80;

    int num_thread = 2;
    float score_threshold = 0.3;
    float nms_threshold = 0.7;
    Params() {}
  } Params;

protected:
  enum nms_type
  {
    hard = 1,
    blending = 2, /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
  };

  typedef struct {
    std::string outputname;
    int stride;
    typedef struct {
      int width;
      int height;
    } Anchor;
    std::vector<Anchor> anchors;
  } Yolov5LayerData;

public:
  ~QGDetector();
  int init(std::string model_path, const Params& params = Params());
  std::vector<BoxInfo> detect(const cv::Mat& frame);

protected:
  std::vector<BoxInfo> decode(MNN::Tensor& data, int stride, std::vector<Yolov5LayerData::Anchor> anchors, int width, int height);
  std::vector<BoxInfo> nms(std::vector<BoxInfo>& inputs, float nms_threshold, int type = nms_type::hard);

private:
  std::shared_ptr<MNN::Interpreter> interpreter = nullptr;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat = nullptr;
  MNN::Session* session = nullptr;
  MNN::Tensor* input_tensor = nullptr;

  bool initialized = false;
  Params params;

  const float norm_vals[3] = { 1.0 / 255, 1.0 / 255, 1.0 / 255 };
  std::vector <Yolov5LayerData> layers =
  {
    {"output", 8,  { {10,  13}, {16,  30},  {33,  23} }},
    {"417",    16, {{30,  61}, {62,  45},  {59,  119}}},
    {"437",    32, {{116, 90}, {156, 198}, {373, 326}}},
  };
};
