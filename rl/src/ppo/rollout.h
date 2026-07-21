#pragma once
#include <torch/torch.h>

#include "vec/vec_env.h"

namespace rl {

struct Rollout {
    Rollout(int T, int A, int obsDim, int criticObsDim, int numBranches,
            bool pinMemory = false);

    int T, A, obsDim, criticObsDim, numBranches;

    torch::Tensor obs;
    torch::Tensor cobs;
    torch::Tensor actions;
    torch::Tensor logp;
    torch::Tensor value;
    torch::Tensor reward;
    torch::Tensor done;

    torch::Tensor truncObs, truncCobs;
    torch::Tensor truncAgent, truncT;
    TruncSink sink;

    void beginRollout();

    std::pair<torch::Tensor, torch::Tensor> computeGAE(const torch::Tensor& lastValue,
                                                       const torch::Tensor& truncValues,
                                                       float gamma, float lam);
};

}
