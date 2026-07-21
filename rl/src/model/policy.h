#pragma once
#include <torch/torch.h>

#include <vector>

#include "config.h"
#include "env/interfaces.h"

namespace rl {

struct PolicyNetImpl : torch::nn::Module {
    PolicyNetImpl(int obsDim, const ActionSpec& spec, const std::vector<int>& hidden,
                  bool layerNorm);

    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs);
    torch::Tensor logitsOnly(const torch::Tensor& obs);

    torch::nn::Sequential trunk{nullptr};
    torch::nn::Linear pi{nullptr}, v{nullptr};
    ActionSpec spec;
    int obsDim;
};
TORCH_MODULE(PolicyNet);

struct CriticNetImpl : torch::nn::Module {
    CriticNetImpl(int obsDim, const std::vector<int>& hidden, bool layerNorm);
    torch::Tensor forward(const torch::Tensor& obs);

    torch::nn::Sequential trunk{nullptr};
    torch::nn::Linear v{nullptr};
    int obsDim;
};
TORCH_MODULE(CriticNet);

std::vector<int> hiddenSizes(const Config& cfg);
PolicyNet buildPolicy(const Config& cfg, int obsDim, const ActionSpec& spec);
CriticNet buildCritic(const Config& cfg, int criticObsDim);

void copyWeights(const torch::nn::Module& src, torch::nn::Module& dst);

}
