#pragma once
#include <functional>
#include <chrono>
#include <future>
#include <cstdio>

class DefferedFunc
{
public:
    template<typename callable, typename... Args>
    DefferedFunc(const int time_ms, const bool use_async, callable&& func, Args&&... args)
    {
        std::function<typename std::result_of<callable(Args...)>::type()> task(std::bind(std::forward<callable>(func),
                                                                                         std::forward<Args>(args)...));
        if (use_async){
            std::thread([time_ms, task](){
                std::this_thread::sleep_for(std::chrono::microseconds(time_ms));
                task();
            }).detach();
        }
        else{
            std::this_thread::sleep_for(std::chrono::microseconds(time_ms));
            task();
        }
    }

};
