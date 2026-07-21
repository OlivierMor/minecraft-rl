#pragma once
#include <torch/torch.h>

#include <memory>

#include "bot.h"
#include "config.h"
#include "env/env.h"
#include "model/policy.h"
#include "replay.h"
#include "vec/worker_pool.h"

namespace rl {

struct MatchController {
    PolicyNet net{nullptr};
    bool deterministic = true;
    bool scripted() const { return net.is_empty(); }
};

struct MatchOutcome {
    int wins = 0, draws = 0, losses = 0;
    int total() const { return wins + draws + losses; }
    double score() const {
        return total() == 0 ? 0.5 : (wins + 0.5 * draws) / (double)total();
    }
};

class MatchRunner {
public:
    MatchRunner(const Config& cfg, int parallel, uint64_t seed, WorkerPool& pool,
                torch::Device device);

    MatchOutcome run(MatchController& A, MatchController& B, int matches,
                     Replay* replayOut = nullptr);

private:
    std::vector<std::unique_ptr<Env>> envs_;
    std::vector<mc::SimpleBot> bots_;
    WorkerPool& workers_;
    torch::Device device_;
    int E_;
    int obsDim_;
    ActionSpec spec_;
    torch::Tensor obsHost_;
    torch::Tensor actionsHost_;
    torch::Tensor rowsA_, rowsB_;
};

}
