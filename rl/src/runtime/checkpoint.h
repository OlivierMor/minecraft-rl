#pragma once
#include <torch/torch.h>

#include <map>
#include <string>

#include "model/policy.h"

namespace rl {

struct TrainState {
    long update = 0;
    long envSteps = 0;
    double elo = 1000;
    std::map<std::string, double> extra;

    void save(const std::string& path) const;
    bool load(const std::string& path);
};

void saveCheckpoint(const std::string& dir, PolicyNet& policy, CriticNet& critic,
                    torch::optim::Adam& opt, const TrainState& st);
bool loadCheckpoint(const std::string& dir, PolicyNet& policy, CriticNet& critic,
                    torch::optim::Adam& opt, TrainState& st, torch::Device device);

bool loadPolicyWeights(const std::string& dir, PolicyNet& policy);

bool saveBestCheckpoint(const std::string& dir, PolicyNet& policy, CriticNet& critic,
                        double elo, long update);

}
