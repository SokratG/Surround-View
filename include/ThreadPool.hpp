#pragma once
#include <functional>
#include <condition_variable>
#include <queue>
#include <thread>
#include <exception>
#include <vector>
#include <atomic>
#include <queue>
#include <future>

#define DISALLOW_COPY_AND_ASSIGN(T) \
    T(const T&) = delete; \
    T &operator=(const T&) = delete;

class ThreadPool
{
public:

	using Task = std::function<void()>;

	explicit ThreadPool(const size_t numThreads) {
		if (numThreads == 0)
                    throw std::runtime_error("invalid input value");

		start(numThreads);
	}
	~ThreadPool() {
		stop();
	}
	
	template<typename T, typename... Args>
	auto enqueue(T&& task, Args&&... args) -> std::future<decltype(task(args...))> {

		std::function<decltype(task(args...))()> func = std::bind(std::forward<T>(task), std::forward<Args>(args)...);

		auto ptr_task = std::make_shared<std::packaged_task<decltype(task(args...)) ()>>(std::move(func));

		std::function<void()> wrapper = [ptr_task]() {
			(*ptr_task)();
		};

		{
			std::unique_lock<std::mutex> lock{ EventMutex };
			Tasks.emplace(wrapper);
		}

		EventVar.notify_all();
		return ptr_task->get_future();
	}


	ThreadPool(const ThreadPool&) = delete;
	ThreadPool(ThreadPool&&) = delete;

	ThreadPool& operator=(const ThreadPool&) = delete;
	ThreadPool& operator=(ThreadPool&&) = delete;

private:
	
	std::queue<Task> Tasks;
	std::vector<std::thread> Threads;
	std::condition_variable EventVar;
	std::mutex EventMutex;
	bool Stopping = false;

	void start(size_t numThreads) {
		for (size_t Ithread = 0u; Ithread < numThreads; ++Ithread) {
			Threads.emplace_back([=]() {
				while (true) {
					Task task;
					{ // scope for unlock mutex before execute
						std::unique_lock<std::mutex> lock{ EventMutex };
						EventVar.wait(lock, [=]() { return Stopping || !Tasks.empty(); });

						if (Stopping && Tasks.empty())
							break;

						task = std::move(Tasks.front());
						Tasks.pop();
					}

					task();

				}
			});
		}
	}
	void stop() noexcept {

		{
			std::unique_lock<std::mutex> lock{ EventMutex };
			Stopping = true;
		}

		EventVar.notify_all();
		for (auto& thread : Threads) {
			thread.join();
		}
	}

};
