#pragma once

#include <future>
#include <functional>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

// thread-local marker to avoid nested deadlock detection
inline thread_local bool g_in_thread_pool_worker = false;

class ThreadPool {
public:
    // singleton accessor
    static ThreadPool& instance();

    // submit task, returns future
    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using R = decltype(f());
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::packaged_task<R()>(std::forward<F>(f)));
        std::future<R> fut = task_ptr->get_future();

        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) throw std::runtime_error("ThreadPool is stopped");
            tasks_.emplace([task_ptr]() mutable {
                g_in_thread_pool_worker = true;
                (*task_ptr)();
                g_in_thread_pool_worker = false;
            });
        }
        cv_.notify_one();
        return fut;
    }

    size_t thread_count() const;

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    ~ThreadPool();

private:
    explicit ThreadPool(size_t threads);

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool done_;
};
