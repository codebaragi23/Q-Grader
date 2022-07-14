#include "classifier.h"

QGClassifier::~QGClassifier()
{
  if (interpreter)
  {
    interpreter->releaseModel();
    interpreter->releaseSession(session);
  }
}

int QGClassifier::init(std::string model_path, const Params& params)
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
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3, norm_vals, 3));

  initialized = true;
  return 1;
};

std::vector<ClassInfo> QGClassifier::classify(const cv::Mat& frame)
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
  MNN::Tensor* tensor = interpreter->getSessionOutput(session, "output");
  MNN::Tensor tensor_host(tensor, tensor->getDimensionType());
  tensor->copyToHostTensor(&tensor_host);
  std::vector<ClassInfo> outputs = decode(tensor_host, frame.cols, frame.rows);
  std::sort(outputs.begin(), outputs.end(), [](const ClassInfo& a, const ClassInfo& b) { return a.score > b.score; });
  return outputs;
}


std::vector<ClassInfo> QGClassifier::decode(MNN::Tensor& data, int width, int height)
{
  std::vector<ClassInfo> outputs;

  int num_classes = std::min(params.num_classes, data.shape()[1]);
  auto data_ptr = data.host<float>();
  for (int id = 0; id < num_classes; id++)
  {
    ClassInfo output;
    output.labelid = id;
    output.score = data_ptr[id];
    outputs.push_back(output);
  }
  return outputs;
}

