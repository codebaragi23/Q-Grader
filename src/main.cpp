#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "argparse.hpp"
#include "timer.hpp"
#include "detector.h"
#include "classifier.h"

#include <vector>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <iostream>

#include <dirent.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/stat.h>

inline std::vector<std::string> getlistdir(const std::string& path, unsigned int st_mode)
{
  std::vector<std::string> list;
  struct dirent* dir;
  DIR* d;
  struct stat st;
  if (d = opendir(path.c_str()))
  {
    while (dir = readdir(d))
    {
      const char* name = dir->d_name;
      if (stat((path + "/" + name).c_str(), &st) == 0 && (st.st_mode & st_mode) && ((st.st_mode & S_IFDIR) != S_IFDIR || (strcmp(name, ".") && strcmp(name, ".."))))
        list.push_back(name);
    }
    closedir(d);
  }
  return list;
}



const char* APP_WINDOW_NAME = "Q-GRADER";

std::vector<std::string> detect_labels =
{
  "COFFEE BEAN", "FOREIGN MATTER",
};

std::vector<std::string> classify_labels =
{
  "BLACK, SOUR", "BROKEN, FISSURE", "EMPTY", "FUNGUS-DAMAGED", "IMMATURE", "INSPECT-DAMAGED",
  "MULTIPLE", "NORMAL",
  "OVER-DRIED, FLOATER", "SHELL", "FOREIGN MATTER",
};

float UD_SCALE = 1.064f;
int UD_TRANS[2] = { 48, 18 };

cv::Rect unionbox(std::vector<cv::Rect> boxes)
{
  int x = boxes[0].x, y = boxes[0].y, w = boxes[0].width, h = boxes[0].height;
  for (auto box : boxes)
  {
    x = std::min(x, box.x);
    y = std::min(y, box.y);
    w = std::max(w, box.width);
    h = std::max(h, box.height);
  }

  return cv::Rect(x, y, w, h);
}

int main(int argc, const char** argv, char* envpp[])
{
  ArgumentParser parser;

  //**** Input ****//
  parser.add_argument("-t", "--input_type", 1, "camera", "camera, video, images");
  parser.add_argument("-i", "--input", 1, "0", "camera id or video file name, images directory", true);
  parser.parse_args(argc, argv);

  std::string type = parser.retrieve<std::string>("input_type");
  std::string input = parser.retrieve<std::string>("input");

  cv::Scalar crDetect(0, 0, 255);

  QGDetector::Params dparams;
  dparams.num_classes = detect_labels.size();
  QGDetector detector;
  detector.init("models/coffee-detector.mnn", dparams);

  QGClassifier::Params cparams;
  cparams.num_classes = classify_labels.size();
  QGClassifier classifier;
  classifier.init("models/coffee-clssifier.mnn", cparams);

  std::vector<std::string> images;
  time_t t = time(NULL);
  struct tm lt = *localtime(&t);
  if (type == "images")
  {
    mkdir("output", 0755);
    std::string outpath = cv::format("output/%04d-%02d-%02d", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday);
    mkdir(outpath.c_str(), 0755);

    std::vector<std::string> subdir = getlistdir(input, S_IFDIR);
    for (std::string dir : subdir)
    {
      mkdir((outpath + "/" + dir).c_str(), 0755);

      std::vector<std::string> imv1 = getlistdir(input + "/" + dir + "/U", S_IFREG);
      std::vector<std::string> imv2 = getlistdir(input + "/" + dir + "/D", S_IFREG);

      std::set<std::string> ims1(std::make_move_iterator(imv1.begin()), std::make_move_iterator(imv1.end()));
      std::set<std::string> ims2(std::make_move_iterator(imv2.begin()), std::make_move_iterator(imv2.end()));
      std::set_intersection(ims1.begin(), ims1.end(), ims2.begin(), ims2.end(), std::inserter(images, images.end()));
      for (std::string name : images)
      {
        Timer::GetInstance().tic();
        cv::Mat imgU = cv::imread(input + "/" + dir + "/U/" + name);
        cv::Mat imgD = cv::imread(input + "/" + dir + "/D/" + name);
        cv::Mat infer = cv::Mat::zeros(imgU.size(), imgU.type());

        cv::resize(imgD, imgD, cv::Size(), UD_SCALE, UD_SCALE);
        imgU = imgU(cv::Rect(0, 0, imgU.cols / 2, imgU.rows));
        imgD = imgD(cv::Rect(UD_TRANS[0], UD_TRANS[1], imgU.cols, imgU.rows));

        Timer::GetInstance().tic();
        std::vector<BoxInfo> udinfos = detector.detect(imgU);
        std::vector<BoxInfo> ddinfos = detector.detect(imgD);
        Timer::GetInstance().toc("    >>> detection: ");

        if (udinfos.size() != ddinfos.size())
        {
          std::vector<BoxInfo> dinfos;
          if (udinfos.size() > 0 && ddinfos.size() > 0)
            dinfos = udinfos;
          else if (udinfos.size() > 0)
            dinfos = udinfos;
          else if (ddinfos.size() > 0)
            dinfos = ddinfos;

          udinfos = dinfos;
          ddinfos = dinfos;
        }

        {
          int w = infer.cols, h = infer.rows;
          {
            std::vector<cv::Rect> boxes;
            for (auto i : udinfos) boxes.push_back(i.bbox);
            cv::Rect box = unionbox(boxes);
            infer(cv::Rect(w / 4 - box.width / 2, h / 2 - box.height / 2, box.width, box.height)) = imgU(box);
            cv::rectangle(imgU, box, crDetect);
          }
          {
            std::vector<cv::Rect> boxes;
            for (auto i : ddinfos) boxes.push_back(i.bbox);
            cv::Rect box = unionbox(boxes);
            infer(cv::Rect(w / 2 + w / 4 - box.width / 2, h / 2 - box.height / 2, box.width, box.height)) = imgD(box);
            cv::rectangle(imgD, box, crDetect);
          }

          Timer::GetInstance().tic();
          auto cinfos = classifier.classify(infer);
          Timer::GetInstance().toc("    >>> classify: ");

          auto labelid = cinfos[0].labelid;
          auto& score = cinfos[0].score;
          auto& label = classify_labels[labelid];
          cv::hconcat(imgU, imgD, infer);
          cv::putText(infer, label, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1, crDetect, 2);
          cv::imwrite(outpath + "/" + dir + "/" + name, infer);
        }

        Timer::GetInstance().toc("total");
      }
    }
  }



  return 0;
}
