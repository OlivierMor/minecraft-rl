#include "model/distributions.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace rl {

namespace {

constexpr float PAD = -1.0e9f;

struct PadMap {
    torch::Tensor idx;
    torch::Tensor mask;
    int64_t maxN = 0;
};

const PadMap& padMap(const ActionSpec& spec, const torch::Device& dev) {
    static std::mutex m;
    static std::unordered_map<std::string, PadMap> cache;
    std::string key = dev.str();
    for (int n : spec.branches) key += "," + std::to_string(n);
    std::lock_guard<std::mutex> lk(m);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    int64_t nb = spec.numBranches();
    int64_t maxN = 0;
    for (int n : spec.branches) maxN = std::max<int64_t>(maxN, n);
    torch::Tensor idx = torch::zeros({nb * maxN}, torch::kInt64);
    torch::Tensor mask = torch::ones({nb * maxN}, torch::kBool);
    auto ia = idx.accessor<int64_t, 1>();
    auto ma = mask.accessor<bool, 1>();
    int64_t off = 0;
    for (int64_t b = 0; b < nb; ++b) {
        int64_t n = spec.branches[(size_t)b];
        for (int64_t j = 0; j < n; ++j) {
            ia[b * maxN + j] = off + j;
            ma[b * maxN + j] = false;
        }
        off += n;
    }
    return cache.emplace(std::move(key), PadMap{idx.to(dev), mask.to(dev), maxN})
        .first->second;
}

torch::Tensor paddedLogSoftmax(const torch::Tensor& logits, const ActionSpec& spec) {
    const PadMap& pm = padMap(spec, logits.device());
    torch::Tensor p = logits.index_select(1, pm.idx).masked_fill(pm.mask, PAD);
    return torch::log_softmax(
        p.view({logits.size(0), (int64_t)spec.numBranches(), pm.maxN}), 2);
}

torch::Tensor gumbelArgmax(const torch::Tensor& ls) {
    torch::Tensor u = torch::rand_like(ls).clamp_min_(1e-20);
    torch::Tensor g = u.log_().neg_().clamp_min_(1e-20).log_().neg_();
    return (ls + g).argmax(2);
}

}

torch::Tensor sampleActions(const torch::Tensor& logits, const ActionSpec& spec,
                            bool deterministic) {
    torch::NoGradGuard ng;
    torch::Tensor ls = paddedLogSoftmax(logits, spec);
    return deterministic ? ls.argmax(2) : gumbelArgmax(ls);
}

std::pair<torch::Tensor, torch::Tensor> sampleWithLogProb(const torch::Tensor& logits,
                                                          const ActionSpec& spec,
                                                          bool deterministic) {
    torch::NoGradGuard ng;
    torch::Tensor ls = paddedLogSoftmax(logits, spec);
    torch::Tensor acts = deterministic ? ls.argmax(2) : gumbelArgmax(ls);
    torch::Tensor lp = ls.gather(2, acts.unsqueeze(2)).squeeze(2).sum(1);
    return {acts, lp};
}

torch::Tensor logProb(const torch::Tensor& logits, const ActionSpec& spec,
                      const torch::Tensor& actions) {
    torch::Tensor ls = paddedLogSoftmax(logits, spec);
    return ls.gather(2, actions.unsqueeze(2)).squeeze(2).sum(1);
}

std::vector<double> branchEntropies(const torch::Tensor& logits, const ActionSpec& spec) {
    torch::NoGradGuard ng;
    torch::Tensor ls = paddedLogSoftmax(logits, spec);
    torch::Tensor ent = (ls.exp() * ls).sum(2).mean(0).neg().to(torch::kCPU);
    auto ea = ent.accessor<float, 1>();
    std::vector<double> out;
    for (int b = 0; b < spec.numBranches(); ++b) out.push_back(ea[b]);
    return out;
}

std::pair<torch::Tensor, torch::Tensor> logProbEntropy(const torch::Tensor& logits,
                                                       const ActionSpec& spec,
                                                       const torch::Tensor& actions) {
    torch::Tensor ls = paddedLogSoftmax(logits, spec);
    torch::Tensor lp = ls.gather(2, actions.unsqueeze(2)).squeeze(2).sum(1);
    torch::Tensor ent = (ls.exp() * ls).sum({1, 2}).neg();
    return {lp, ent};
}

}
