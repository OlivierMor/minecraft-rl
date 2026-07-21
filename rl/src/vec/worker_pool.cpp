#include "vec/worker_pool.h"

namespace rl {

WorkerPool::WorkerPool(int workers) {
    if (workers < 1) workers = 1;
    threads_.reserve((size_t)workers);
    for (int i = 0; i < workers; ++i) threads_.emplace_back([this, i] { loop(i); });
}

WorkerPool::~WorkerPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cvJob_.notify_all();
    for (auto& t : threads_) t.join();
}

void WorkerPool::run(const std::function<void(int)>& fn) {
    std::unique_lock<std::mutex> lk(m_);
    job_ = &fn;
    pending_ = (int)threads_.size();
    ++generation_;
    cvJob_.notify_all();
    cvDone_.wait(lk, [this] { return pending_ == 0; });
    job_ = nullptr;
}

void WorkerPool::loop(int idx) {
    long seen = 0;
    while (true) {
        const std::function<void(int)>* job;
        {
            std::unique_lock<std::mutex> lk(m_);
            cvJob_.wait(lk, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            job = job_;
        }
        (*job)(idx);
        {
            std::lock_guard<std::mutex> lk(m_);
            if (--pending_ == 0) cvDone_.notify_all();
        }
    }
}

}
