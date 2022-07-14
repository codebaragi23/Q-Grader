#include "detector.h"

QGDetector::~QGDetector()
{
  if (interpreter)
  {
    interpreter->releaseModel();
    interpreter->releaseSession(session);
  }
}

int QGDetector::init(std::string model_path, const Params& params)
{
  interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
  if (interpreter == nullptr) return 0;

  this->params = params;
  MNN::ScheduleConfig schedule_config;
  schedule_config.numThread = params.num_thread;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_Low;
  schedule_config.backendConfig = &backend_config;

  session = interpreter->createSession(schedule_config);
  input_tensor = interpreter->getSessionInput(session, nullptr);

  interpreter->resizeTensor(input_tensor, { 1, params.channel, params.height, params.width });
  interpreter->resizeSession(session);
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, nullptr, 0, norm_vals, 3));

  initialized = true;
  return 1;
};

std::vector<BoxInfo> QGDetector::detect(const cv::Mat& frame)
{
  if (!initialized)
  {
    fprintf(stderr, "(!)----Error: model uninitialized.\n");
    return {};
  }
  if (frame.empty())
  {
    fprintf(stderr, "(!)----Error: image is empty, please check!\n");
    return {};
  }

  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(params.width, params.height));
  pretreat->convert(resized.data, params.width, params.height, resized.step[0], input_tensor);

  // run network
  interpreter->runSession(session);

  // get output data
  std::vector<BoxInfo> boxes;
  for (auto layer : layers)
  {
    MNN::Tensor* tensor = interpreter->getSessionOutput(session, layer.outputname.c_str());
    MNN::Tensor tensor_host(tensor, tensor->getDimensionType());
    tensor->copyToHostTensor(&tensor_host);
    std::vector<BoxInfo> outputs = decode(tensor_host, layer.stride, layer.anchors, frame.cols, frame.rows);
    boxes.insert(boxes.end(), outputs.begin(), outputs.end());
  }
  return nms(boxes, params.nms_threshold);
}


inline float fast_exp(float x)
{
  union
  {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}
inline float sigmoid(float x)
{
  return 1.0f / (1.0f + fast_exp(-x));
}
std::vector<BoxInfo> QGDetector::decode(MNN::Tensor& data, int stride, std::vector<Yolov5LayerData::Anchor> anchors, int width, int height)
{
  std::vector<BoxInfo> outputs;

  int batch = data.shape()[0];
  int channels = data.shape()[1];
  int dh = data.shape()[2];
  int dw = data.shape()[3];
  int preds = data.shape()[4];

  auto data_ptr = data.host<float>();
  for (int b = 0; b < batch; b++)
  {
    auto batch_ptr = data_ptr + b * (channels * dh * dw * preds);
    for (int c = 0; c < channels; c++)
    {
      auto channel_ptr = batch_ptr + c * (dh * dw * preds);
      for (int h = 0; h < dh; h++)
      {
        auto height_ptr = channel_ptr + h * (dw * preds);
        for (int w = 0; w < dw; w++)
        {
          auto width_ptr = height_ptr + w * preds;
          auto cls_ptr = width_ptr + 5;

          auto confidence = sigmoid(width_ptr[4]);

          for (int id = 0; id < params.num_classes; id++)
          {
            float score = sigmoid(cls_ptr[id]) * confidence;
            if (score > params.score_threshold)
            {
              float cx = (sigmoid(width_ptr[0]) * 2.f - 0.5f + w) * (float)stride / params.width;
              float cy = (sigmoid(width_ptr[1]) * 2.f - 0.5f + h) * (float)stride / params.height;
              float sw = pow(sigmoid(width_ptr[2]) * 2.f, 2) * anchors[c].width / params.width;
              float sh = pow(sigmoid(width_ptr[3]) * 2.f, 2) * anchors[c].height / params.height;

              BoxInfo output;
              output.bbox.x = (cx - sw / 2.f) * width;
              output.bbox.y = (cy - sh / 2.f) * height;
              output.bbox.width = sw * width;
              output.bbox.height = sh * height;
              output.bbox &= cv::Rect(0, 0, width, height);
              output.score = score;
              output.labelid = id;
              outputs.push_back(output);
            }
          }
        }
      }
    }
  }

  return outputs;
}

std::vector<BoxInfo> QGDetector::nms(std::vector<BoxInfo>& inputs, float nms_threshold, int type)
{
  std::vector<BoxInfo> outputs;
  std::sort(inputs.begin(), inputs.end(), [](const BoxInfo& a, const BoxInfo& b) { return a.score > b.score; });
  int box_num = inputs.size();
  std::vector<int> merged(box_num, 0);

  for (int i = 0; i < box_num; i++)
  {
    if (merged[i])
      continue;
    std::vector<BoxInfo> outs;

    outs.push_back(inputs[i]);
    merged[i] = 1;

    float area0 = inputs[i].bbox.width * inputs[i].bbox.height;

    for (int j = i + 1; j < box_num; j++)
    {
      if (merged[j])
        continue;

      float inner_x0 = std::max(inputs[i].bbox.x, inputs[j].bbox.x);
      float inner_y0 = std::max(inputs[i].bbox.y, inputs[j].bbox.y);

      float inner_x1 = std::min(inputs[i].bbox.br().x, inputs[j].bbox.br().x);
      float inner_y1 = std::min(inputs[i].bbox.br().y, inputs[j].bbox.br().y);

      float inner_w = inner_x1 - inner_x0 + 1;
      float inner_h = inner_y1 - inner_y0 + 1;

      if (inner_h <= 0 || inner_w <= 0)
        continue;

      float inner_area = inner_h * inner_w;
      float area1 = inputs[j].bbox.width * inputs[j].bbox.height;
      float score = inner_area / (area0 + area1 - inner_area);
      if (score > params.nms_threshold)
      {
        merged[j] = 1;
        outs.push_back(inputs[j]);
      }
    }
    switch (type)
    {
    case nms_type::hard:
    {
      outputs.push_back(outs[0]);
      break;
    }
    case nms_type::blending:
    {
      float total = 0;
      for (int i = 0; i < outs.size(); i++)
      {
        total += exp(outs[i].score);
      }
      BoxInfo out;
      memset(&out, 0, sizeof(out));
      for (int i = 0; i < outs.size(); i++)
      {
        float rate = exp(outs[i].score) / total;
        out.bbox.x += outs[i].bbox.x * rate;
        out.bbox.y += outs[i].bbox.y * rate;
        out.bbox.width += outs[i].bbox.width * rate;
        out.bbox.height += outs[i].bbox.height * rate;
        out.score += outs[i].score * rate;
      }
      outputs.push_back(out);
      break;
    }
    default:
    {
      fprintf(stderr, "(!)----Error: Wrong type of nms.");
      exit(-1);
    }
    }
  }

  return outputs;
}
