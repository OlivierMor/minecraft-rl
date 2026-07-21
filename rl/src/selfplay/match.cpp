#include "selfplay/match.h"

#include <algorithm>
#include <atomic>

#include "env/registry.h"
#include "model/distributions.h"

namespace rl {

MatchRunner::MatchRunner(const Config& cfg, int parallel, uint64_t seed, WorkerPool& pool,
                         torch::Device device)
    : workers_(pool), device_(device), E_(parallel) {
    envs_.reserve((size_t)E_);
    for (int i = 0; i < E_; ++i)
        envs_.push_back(std::make_unique<Env>(i, seed + 77777ULL + (uint64_t)i * 131ULL,
                                              registry::makeComponents(cfg)));
    bots_.resize((size_t)E_);
    float turn = (float)cfg.num("bot.max_turn", 40.0);
    int click = (int)cfg.num("bot.click_interval", 3);
    for (auto& b : bots_) {
        b.maxTurnPerTick = turn;
        b.clickInterval = click;
    }
    obsDim_ = envs_[0]->obsSize();
    spec_ = envs_[0]->spec();
    obsHost_ = torch::zeros({2 * E_, obsDim_}, torch::kFloat32);
    actionsHost_ = torch::zeros({2 * E_, spec_.numBranches()}, torch::kInt32);
    torch::Tensor ra = torch::zeros({E_}, torch::kInt64);
    torch::Tensor rb = torch::zeros({E_}, torch::kInt64);
    for (int e = 0; e < E_; ++e) {
        int sa = e % 2;
        ra[e] = sa * E_ + e;
        rb[e] = (1 - sa) * E_ + e;
    }
    rowsA_ = ra;
    rowsB_ = rb;
}

MatchOutcome MatchRunner::run(MatchController& A, MatchController& B, int matches,
                              Replay* replayOut) {
    for (auto& e : envs_) e->reset();
    if (!A.scripted()) A.net->eval();
    if (!B.scripted()) B.net->eval();

    std::vector<MatchOutcome> perWorker((size_t)workers_.size());
    std::vector<std::array<mc::Input, 2>> replayTicks;
    std::vector<SetupOp> replaySetup = envs_[0]->setupOps;
    bool replayDone = replayOut == nullptr;

    float* obsPtr = obsHost_.data_ptr<float>();
    int32_t* actPtr = actionsHost_.data_ptr<int32_t>();
    const int nb = spec_.numBranches();

    auto controllerAt = [&](int env, int side) -> MatchController& {
        return (env % 2) == side ? A : B;
    };

    int total = 0;
    long guard = 0;
    while (total < matches && guard++ < 1000000) {
        std::atomic<int> obsCursor{0};
        workers_.run([&](int) {
            for (int b; (b = obsCursor.fetch_add(4, std::memory_order_relaxed)) < E_;)
                for (int e = b, e1 = std::min(E_, b + 4); e < e1; ++e) {
                    envs_[(size_t)e]->buildObs(0, obsPtr + (size_t)e * obsDim_);
                    envs_[(size_t)e]->buildObs(1, obsPtr + ((size_t)E_ + e) * obsDim_);
                }
        });

        {
            torch::NoGradGuard ng;
            torch::Tensor obsDev = obsHost_.to(device_);
            for (int c = 0; c < 2; ++c) {
                MatchController& ctl = c == 0 ? A : B;
                if (ctl.scripted()) continue;
                const torch::Tensor& rows = c == 0 ? rowsA_ : rowsB_;
                torch::Tensor logits = ctl.net->logitsOnly(obsDev.index_select(0, rows.to(device_)));
                torch::Tensor acts =
                    sampleActions(logits, spec_, ctl.deterministic).to(torch::kCPU, torch::kInt32);
                auto ar = acts.accessor<int32_t, 2>();
                auto rr = rows.accessor<int64_t, 1>();
                for (int e = 0; e < E_; ++e)
                    for (int b = 0; b < nb; ++b) actPtr[rr[e] * nb + b] = ar[e][b];
            }
        }

        std::atomic<int> stepCursor{0};
        workers_.run([&](int w) {
            for (int b; (b = stepCursor.fetch_add(4, std::memory_order_relaxed)) < E_;)
                for (int e = b, e1 = std::min(E_, b + 4); e < e1; ++e) {
                    Env& env = *envs_[(size_t)e];
                    mc::Input in[2];
                    for (int s = 0; s < 2; ++s) {
                        if (controllerAt(e, s).scripted())
                            in[s] = bots_[(size_t)e].act(env.sim.client[s], env.sim.tickCount);
                        else
                            in[s] = env.parseAction(s, actPtr + (size_t)(s * E_ + e) * nb);
                    }
                    if (e == 0 && !replayDone) replayTicks.push_back({in[0], in[1]});
                    Env::StepOut out = env.step(in[0], in[1]);
                    if (out.done) {
                        int aSide = e % 2;
                        if (out.winner < 0) ++perWorker[(size_t)w].draws;
                        else if (out.winner == aSide) ++perWorker[(size_t)w].wins;
                        else ++perWorker[(size_t)w].losses;
                        env.reset();
                    }
                }
        });

        if (!replayDone) {
            if (envs_[0]->sim.tickCount < (int)replayTicks.size()) {
                replayOut->setup = replaySetup;
                replayOut->ticks = replayTicks;
                replayDone = true;
            }
        }

        total = 0;
        for (const auto& o : perWorker) total += o.total();
    }

    MatchOutcome sum;
    for (const auto& o : perWorker) {
        sum.wins += o.wins;
        sum.draws += o.draws;
        sum.losses += o.losses;
    }
    return sum;
}

}
