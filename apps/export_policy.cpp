#include <torch/torch.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "config.h"
#include "env/registry.h"
#include "model/policy.h"
#include "runtime/checkpoint.h"

namespace fs = std::filesystem;
using namespace rl;

namespace {

void writeI32(std::ofstream& f, int32_t v) { f.write((const char*)&v, 4); }

void writeF32(std::ofstream& f, float v) { f.write((const char*)&v, 4); }

void writeTensor(std::ofstream& f, const torch::Tensor& t) {
    torch::Tensor c = t.contiguous().to(torch::kFloat32);
    f.write((const char*)c.data_ptr<float>(), c.numel() * 4);
}

}

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::fprintf(stderr, "usage: export <run_dir|policy.pt> <out.rlw> [--config c]\n");
        return 1;
    }
    std::string ref = argv[1], out = argv[2], cfgPath;
    for (int i = 3; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--config") cfgPath = argv[i + 1];
    if (cfgPath.empty() && fs::is_directory(ref) && fs::exists(ref + "/config.toml"))
        cfgPath = ref + "/config.toml";
    if (cfgPath.empty()) cfgPath = MC_ENV_DEFAULT_CONFIG;
    Config cfg = Config::fromFile(cfgPath);

    auto obs = registry::makeObs(cfg.str("components.obs", "standard"), cfg, 1);
    auto parser = registry::makeParser(cfg.str("components.actions", "standard"), cfg);
    ActionSpec spec = parser->spec();
    std::vector<int> hidden = hiddenSizes(cfg);
    bool layerNorm = cfg.boolean("model.layernorm", true);

    PolicyNet net = buildPolicy(cfg, obs->size(), spec);
    if (!loadPolicyWeights(ref, net))
        throw std::runtime_error("cannot load policy from: " + ref);
    net->eval();

    std::ofstream f(out, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open output: " + out);
    f.write("RLWA", 4);
    writeI32(f, obs->size());
    writeI32(f, spec.numBranches());
    for (int b : spec.branches) writeI32(f, b);
    writeI32(f, (int32_t)hidden.size());
    for (int h : hidden) writeI32(f, h);
    writeI32(f, layerNorm ? 1 : 0);

    std::vector<double> yawB = cfg.numArr("actions.yaw_buckets");
    std::vector<double> pitchB = cfg.numArr("actions.pitch_buckets");
    if (yawB.empty()) yawB = {-45, -20, -10, -4, -1, 0, 1, 4, 10, 20, 45};
    if (pitchB.empty()) pitchB = {-20, -8, -3, -1, 0, 1, 3, 8, 20};
    writeI32(f, (int32_t)yawB.size());
    for (double v : yawB) writeF32(f, (float)v);
    writeI32(f, (int32_t)pitchB.size());
    for (double v : pitchB) writeF32(f, (float)v);

    for (const auto& child : net->trunk->children()) {
        if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(child)) {
            writeTensor(f, lin->weight);
            writeTensor(f, lin->bias);
        } else if (auto ln = std::dynamic_pointer_cast<torch::nn::LayerNormImpl>(child)) {
            writeTensor(f, ln->weight);
            writeTensor(f, ln->bias);
        }
    }
    writeTensor(f, net->pi->weight);
    writeTensor(f, net->pi->bias);
    f.close();

    torch::Tensor x = torch::zeros({1, obs->size()});
    for (int i = 0; i < obs->size(); ++i)
        x[0][i] = std::sin(0.7 * i) * 0.5;
    torch::NoGradGuard ng;
    torch::Tensor logits = net->logitsOnly(x)[0];
    std::printf("exported %s -> %s\n", ref.c_str(), out.c_str());
    std::printf("obs_dim=%d branches=", obs->size());
    for (int b : spec.branches) std::printf("%d ", b);
    std::printf("\nself-test logits (obs[i]=sin(0.7i)*0.5):\n");
    int off = 0;
    for (int bi = 0; bi < spec.numBranches(); ++bi) {
        int n = spec.branches[(size_t)bi];
        int best = 0;
        std::printf("  b%d:", bi);
        for (int j = 0; j < n; ++j) {
            float v = logits[off + j].item<float>();
            std::printf(" %.6f", v);
            if (v > logits[off + best].item<float>()) best = j;
        }
        std::printf("  argmax=%d\n", best);
        off += n;
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
}
