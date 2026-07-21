#include "ppo/rollout.h"

namespace rl {

Rollout::Rollout(int T_, int A_, int obsDim_, int criticObsDim_, int numBranches_,
                 bool pinMemory)
    : T(T_), A(A_), obsDim(obsDim_), criticObsDim(criticObsDim_), numBranches(numBranches_) {
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(pinMemory);
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(pinMemory);
    obs = torch::zeros({T, A, obsDim}, f32);
    if (criticObsDim > 0) cobs = torch::zeros({T, A, criticObsDim}, f32);
    actions = torch::zeros({T, A, numBranches}, i32);
    logp = torch::zeros({T, A}, f32);
    value = torch::zeros({T, A}, f32);
    reward = torch::zeros({T, A}, f32);
    done = torch::zeros({T, A}, torch::kUInt8);

    int cap = 4 * A;
    truncObs = torch::zeros({cap, obsDim}, f32);
    if (criticObsDim > 0) truncCobs = torch::zeros({cap, criticObsDim}, f32);
    truncAgent = torch::zeros({cap}, torch::kInt32);
    truncT = torch::zeros({cap}, torch::kInt32);
    sink.obs = truncObs.data_ptr<float>();
    sink.cobs = criticObsDim > 0 ? truncCobs.data_ptr<float>() : nullptr;
    sink.agent = truncAgent.data_ptr<int32_t>();
    sink.t = truncT.data_ptr<int32_t>();
    sink.cap = cap;
}

void Rollout::beginRollout() {
    sink.n.store(0, std::memory_order_relaxed);
    sink.dropped.store(0, std::memory_order_relaxed);
}

std::pair<torch::Tensor, torch::Tensor> Rollout::computeGAE(const torch::Tensor& lastValue,
                                                            const torch::Tensor& truncValues,
                                                            float gamma, float lam) {
    int nTrunc = std::min(sink.n.load(), sink.cap);
    if (nTrunc > 0) {
        auto rw = reward.accessor<float, 2>();
        auto tv = truncValues.accessor<float, 1>();
        auto ag = truncAgent.accessor<int32_t, 1>();
        auto tt = truncT.accessor<int32_t, 1>();
        for (int k = 0; k < nTrunc; ++k) rw[tt[k]][ag[k]] += gamma * tv[k];
    }

    torch::Tensor adv = torch::zeros({T, A}, torch::kFloat32);
    auto advA = adv.accessor<float, 2>();
    auto rw = reward.accessor<float, 2>();
    auto va = value.accessor<float, 2>();
    auto dn = done.accessor<uint8_t, 2>();
    auto lv = lastValue.accessor<float, 1>();

    std::vector<float> running((size_t)A, 0.0f);
    for (int t = T - 1; t >= 0; --t) {
        for (int a = 0; a < A; ++a) {
            float nonterm = dn[t][a] ? 0.0f : 1.0f;
            float nextV = t == T - 1 ? lv[a] : va[t + 1][a];
            float delta = rw[t][a] + gamma * nonterm * nextV - va[t][a];
            running[(size_t)a] = delta + gamma * lam * nonterm * running[(size_t)a];
            advA[t][a] = running[(size_t)a];
        }
    }
    return {adv, adv + value};
}

}
