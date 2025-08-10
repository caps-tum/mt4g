#include "typedef/threadPool.hpp"

ThreadPool& ThreadPool::instance() {
    static ThreadPool tp(std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
    return tp;
}

size_t ThreadPool::thread_count() const {
    return workers_.size();
}

ThreadPool::ThreadPool(size_t threads) : done_(false) {
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [this]() { return done_ || !tasks_.empty(); });
                    if (done_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}
