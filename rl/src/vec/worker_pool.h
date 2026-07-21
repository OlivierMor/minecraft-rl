#pragma once
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace rl {

class WorkerPool {
public:
    explicit WorkerPool(int workers);
    ~WorkerPool();
    WorkerPool(const WorkerPool&) = delete;

    void run(const std::function<void(int)>& fn);
    int size() const { return (int)threads_.size(); }

    static std::pair<int, int> slice(int total, int w, int n) {
        int per = total / n, rem = total % n;
        int begin = w * per + (w < rem ? w : rem);
        return {begin, begin + per + (w < rem ? 1 : 0)};
    }

private:
    void loop(int idx);

    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cvJob_, cvDone_;
    const std::function<void(int)>* job_ = nullptr;
    long generation_ = 0;
    int pending_ = 0;
    bool stop_ = false;
};

}
