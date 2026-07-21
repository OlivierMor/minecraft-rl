#include "model/policy.h"

namespace rl {

namespace {

torch::nn::Sequential makeTrunk(int in, const std::vector<int>& hidden, bool layerNorm) {
    torch::nn::Sequential t;
    int d = in;
    for (int h : hidden) {
        auto lin = torch::nn::Linear(d, h);
        torch::nn::init::orthogonal_(lin->weight, std::sqrt(2.0));
        torch::nn::init::zeros_(lin->bias);
        t->push_back(lin);
        if (layerNorm) t->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({h})));
        t->push_back(torch::nn::SiLU());
        d = h;
    }
    return t;
}

torch::nn::Linear makeHead(int in, int out, double gain) {
    auto lin = torch::nn::Linear(in, out);
    torch::nn::init::orthogonal_(lin->weight, gain);
    torch::nn::init::zeros_(lin->bias);
    return lin;
}

}

PolicyNetImpl::PolicyNetImpl(int obsDim, const ActionSpec& spec_,
                             const std::vector<int>& hidden, bool layerNorm)
    : spec(spec_), obsDim(obsDim) {
    trunk = register_module("trunk", makeTrunk(obsDim, hidden, layerNorm));
    int last = hidden.empty() ? obsDim : hidden.back();
    pi = register_module("pi", makeHead(last, spec.totalLogits(), 0.01));
    v = register_module("v", makeHead(last, 1, 1.0));
}

std::pair<torch::Tensor, torch::Tensor> PolicyNetImpl::forward(const torch::Tensor& obs) {
    torch::Tensor h = trunk->forward(obs);
    return {pi->forward(h), v->forward(h).squeeze(-1)};
}

torch::Tensor PolicyNetImpl::logitsOnly(const torch::Tensor& obs) {
    return pi->forward(trunk->forward(obs));
}

CriticNetImpl::CriticNetImpl(int obsDim, const std::vector<int>& hidden, bool layerNorm)
    : obsDim(obsDim) {
    trunk = register_module("trunk", makeTrunk(obsDim, hidden, layerNorm));
    int last = hidden.empty() ? obsDim : hidden.back();
    v = register_module("v", makeHead(last, 1, 1.0));
}

torch::Tensor CriticNetImpl::forward(const torch::Tensor& obs) {
    return v->forward(trunk->forward(obs)).squeeze(-1);
}

std::vector<int> hiddenSizes(const Config& cfg) {
    auto v = cfg.numArr("model.trunk");
    if (v.empty()) return {512, 512};
    std::vector<int> out;
    for (double d : v) out.push_back((int)d);
    return out;
}

PolicyNet buildPolicy(const Config& cfg, int obsDim, const ActionSpec& spec) {
    return PolicyNet(obsDim, spec, hiddenSizes(cfg), cfg.boolean("model.layernorm", true));
}

CriticNet buildCritic(const Config& cfg, int criticObsDim) {
    return CriticNet(criticObsDim, hiddenSizes(cfg), cfg.boolean("model.layernorm", true));
}

void copyWeights(const torch::nn::Module& src, torch::nn::Module& dst) {
    torch::NoGradGuard ng;
    auto sp = src.parameters();
    auto dp = dst.parameters();
    TORCH_CHECK(sp.size() == dp.size(), "copyWeights: architecture mismatch");
    for (size_t i = 0; i < sp.size(); ++i) dp[i].copy_(sp[i]);
    auto sb = src.buffers();
    auto db = dst.buffers();
    for (size_t i = 0; i < sb.size(); ++i) db[i].copy_(sb[i]);
}

}
