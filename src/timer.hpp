#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <stack>
#include <chrono>

class Timer
{
private:
  std::stack<std::chrono::high_resolution_clock::time_point> tictoc_stack;

public:
  static Timer& GetInstance()
  {
    static Timer timer;
    return timer;
  }

  void tic()
  {
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    tictoc_stack.push(t0);
  }

  double toc(std::string msg = "", bool show = true)
  {
    if (tictoc_stack.empty())
      return 0.0;

    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - tictoc_stack.top());
    double elapsed = 1000 * duration.count() + std::numeric_limits<double>::epsilon();
    if (msg.size() > 0 && show)
      printf("%s time elapsed: %f ms\n", msg.c_str(), elapsed);

    tictoc_stack.pop();
    return elapsed;
  }

  double update()
  {
    if (tictoc_stack.empty())
      return 0.0;

    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - tictoc_stack.top());
    double elapsed = 1000 * duration.count() + std::numeric_limits<double>::epsilon();
    return elapsed;
  }

  void reset()
  {
    tictoc_stack = std::stack<std::chrono::high_resolution_clock::time_point>();
  }
};

#endif //TIMER_H