#pragma once
#include <torch/torch.h>

#include "config.h"
#include "model/policy.h"
#include "ppo/rollout.h"

namespace rl {

struct PPOStats {
    double piLoss = 0, vLoss = 0, entropy = 0, kl = 0, clipFrac = 0;
    double maxLogit = 0;
    std::vector<double> entBranch;
    int epochsRan = 0;
};

struct PPOParams {
    float gamma, lam, clip, vfCoef, gradClip, klStop, vClip;
    float dualClip;
    float logitL2;
    int epochs, minibatch;
    bool normAdv;
    float lr;
    float entCoef;

    static PPOParams fromConfig(const Config& cfg);
};

class Learner {
public:
    Learner(PolicyNet policy, CriticNet critic, torch::Device device, const Config& cfg);

    PPOStats update(Rollout& roll, const torch::Tensor& adv, const torch::Tensor& ret,
                    const PPOParams& p);

    void setLr(double lr);
    torch::optim::Adam& optimizer() { return *opt_; }
    PolicyNet& policy() { return policy_; }
    CriticNet& critic() { return critic_; }
    bool hasCritic() const { return hasCritic_; }
    torch::Device device() const { return device_; }

private:
    PolicyNet policy_;
    CriticNet critic_;
    bool hasCritic_;
    torch::Device device_;
    std::unique_ptr<torch::optim::Adam> opt_;
    std::vector<torch::Tensor> allParams_;
    bool gpuBatch_;
    bool bf16_;
};

}
