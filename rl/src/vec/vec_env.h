#pragma once
#include <atomic>
#include <memory>
#include <vector>

#include "env/sim_api.h"
#include "config.h"
#include "env/env.h"
#include "vec/worker_pool.h"

namespace rl {

struct EpisodeStat {
    int env = 0;
    int8_t oppKind = 0;
    int16_t oppGroup = -1;
    int8_t winner = -1;
    bool truncated = false;
    int ticks = 0;
    int hits[2] = {0, 0};
    int crits[2] = {0, 0};
    int jumps[2] = {0, 0};
    int maxCombo[2] = {0, 0};
};

struct TruncSink {
    float* obs = nullptr;
    float* cobs = nullptr;
    int32_t* agent = nullptr;
    int32_t* t = nullptr;
    int cap = 0;
    std::atomic<int> n{0};
    std::atomic<int> dropped{0};
};

class VecEnv {
public:
    VecEnv(const Config& cfg, int numEnvs, uint64_t seed, WorkerPool& pool);

    void setPartition(int mirror, int poolEnvs, int scripted, int poolGroups);

    int N() const { return (int)envs_.size(); }
    int M() const { return mirror_; }
    int P() const { return poolEnvs_; }
    int A() const { return N() + M(); }
    int R() const { return N() + M() + P(); }
    int obsDim() const { return obsDim_; }
    int criticObsDim() const { return criticObsDim_; }
    const ActionSpec& spec() const { return spec_; }

    std::pair<int, int> poolGroupRows(int g) const {
        return {A() + groupStart_[(size_t)g] - mirror_, A() + groupEnd_[(size_t)g] - mirror_};
    }
    std::pair<int, int> poolGroupOppRows(int g) const {
        return {groupStart_[(size_t)g] - mirror_, groupEnd_[(size_t)g] - mirror_};
    }
    int poolGroups() const { return (int)groupStart_.size(); }

    void buildObs(float* learnerOut, float* oppOut, float* criticOut,
                  int e0 = 0, int e1 = -1);


    void step(const int32_t* actions, float* rewardOut, uint8_t* doneOut,
              TruncSink* trunc, int tStep, int e0 = 0, int e1 = -1);

    void stepSerial(const int32_t* actions, float* rewardOut, uint8_t* doneOut,
                    TruncSink* trunc, int tStep);

    std::vector<EpisodeStat> drainStats();

    Env& env(int i) { return *envs_[(size_t)i]; }
    std::vector<std::pair<std::string, double>> rewardStats();
    void onUpdate(long update) {
        for (auto& e : envs_) e->rewardFn().onUpdate(update);
    }

private:
    void stepRange(int e0, int e1, const int32_t* actions, float* rewardOut,
                   uint8_t* doneOut, TruncSink* trunc, int tStep, int worker);

    std::vector<std::unique_ptr<Env>> envs_;
    std::vector<mc::SimpleBot> bots_;
    WorkerPool& workers_;
    int mirror_ = 0, poolEnvs_ = 0, scripted_ = 0;
    std::vector<int> groupStart_, groupEnd_;
    std::vector<int16_t> groupOfEnv_;
    std::vector<std::vector<EpisodeStat>> workerStats_;
    int obsDim_ = 0, criticObsDim_ = 0;
    ActionSpec spec_;
};

}
