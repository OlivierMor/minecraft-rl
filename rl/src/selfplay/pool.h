#pragma once
#include <torch/torch.h>

#include <string>
#include <vector>

#include "config.h"
#include "model/policy.h"

namespace rl {

struct Snapshot {
    int id = 0;
    long bornUpdate = 0;
    double elo = 1000;
    double winEma = 0.5;
    long games = 0;
    std::vector<torch::Tensor> params;
};

class OpponentPool {
public:
    explicit OpponentPool(const Config& cfg);

    bool empty() const { return members_.empty(); }
    int size() const { return (int)members_.size(); }
    const std::vector<Snapshot>& members() const { return members_; }
    Snapshot* find(int id);

    int snapshot(const PolicyNet& net, double currentElo, long update, const std::string& dir);

    int samplePFSP(std::mt19937_64& rng) const;
    void loadInto(int id, PolicyNet& replica) const;
    void recordResult(int id, double learnerScore);

    void saveMeta(const std::string& dir) const;
    void load(const std::string& dir);

private:
    void prune();

    std::vector<Snapshot> members_;
    int nextId_ = 0;
    int cap_;
    double pfspP_, uniformMix_, emaAlpha_;
};

}
