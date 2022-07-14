#pragma once

#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"

#include <opencv2/opencv.hpp>

typedef struct
{
  int   labelid;
  float score;
} ClassInfo;

class QGClassifier
{
public:
  typedef struct Params
  {
    int width = 224;
    int height = 224;
    int channel = 3;

    int num_classes = 1000;

    int num_thread = 2;
    Params() {}
  } Params;

  enum nms_type
  {
    hard = 1,
    blending = 2, /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
  };

public:
  ~QGClassifier();
  int init(std::string model_path, const Params& params = Params());
  std::vector<ClassInfo> classify(const cv::Mat& frame);

protected:
  std::vector<ClassInfo> decode(MNN::Tensor& data, int width, int height);

private:
  std::shared_ptr<MNN::Interpreter> interpreter = nullptr;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat = nullptr;
  MNN::Session* session = nullptr;
  MNN::Tensor* input_tensor = nullptr;

  bool initialized = false;
  Params params;

  const float mean_vals[3] = { 127.5, 127.5, 127.5 };
  const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };
};
