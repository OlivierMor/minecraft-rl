#include "selfplay/pool.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace rl {

namespace {
std::vector<torch::Tensor> cloneAllToCpu(const torch::nn::Module& m) {
    torch::NoGradGuard ng;
    std::vector<torch::Tensor> out;
    for (const auto& p : m.parameters()) out.push_back(p.detach().to(torch::kCPU).clone());
    for (const auto& b : m.buffers()) out.push_back(b.detach().to(torch::kCPU).clone());
    return out;
}
std::string snapPath(const std::string& dir, int id) {
    return dir + "/pool/snap_" + std::to_string(id) + ".pt";
}
}

OpponentPool::OpponentPool(const Config& cfg) {
    cap_ = (int)cfg.num("selfplay.pool_cap", 32);
    pfspP_ = cfg.num("selfplay.pfsp_p", 2.0);
    uniformMix_ = cfg.num("selfplay.pfsp_uniform", 0.15);
    emaAlpha_ = cfg.num("selfplay.winrate_ema", 0.05);
}

Snapshot* OpponentPool::find(int id) {
    for (auto& s : members_)
        if (s.id == id) return &s;
    return nullptr;
}

int OpponentPool::snapshot(const PolicyNet& net, double currentElo, long update,
                           const std::string& dir) {
    Snapshot s;
    s.id = nextId_++;
    s.bornUpdate = update;
    s.elo = currentElo;
    s.params = cloneAllToCpu(*net);
    fs::create_directories(dir + "/pool");
    torch::save(s.params, snapPath(dir, s.id));
    members_.push_back(std::move(s));
    prune();
    saveMeta(dir);
    return members_.back().id;
}

void OpponentPool::prune() {
    if ((int)members_.size() <= cap_) return;
    const int keepNewest = 4;
    while ((int)members_.size() > cap_) {
        size_t candEnd = members_.size() - (size_t)keepNewest;
        std::vector<size_t> idx;
        for (size_t i = 0; i < candEnd; ++i) idx.push_back(i);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return members_[a].elo < members_[b].elo; });
        size_t drop = idx[0];
        double best = 1e18;
        for (size_t k = 0; k < idx.size(); ++k) {
            double gap = 1e18;
            if (k > 0) gap = std::min(gap, members_[idx[k]].elo - members_[idx[k - 1]].elo);
            if (k + 1 < idx.size()) gap = std::min(gap, members_[idx[k + 1]].elo - members_[idx[k]].elo);
            if (gap < best) {
                best = gap;
                drop = idx[k];
            }
        }
        members_.erase(members_.begin() + (long)drop);
    }
}

int OpponentPool::samplePFSP(std::mt19937_64& rng) const {
    std::vector<double> w(members_.size());
    double total = 0;
    for (size_t i = 0; i < members_.size(); ++i) {
        w[i] = std::pow(1.0 - members_[i].winEma, pfspP_) + uniformMix_;
        total += w[i];
    }
    double r = std::uniform_real_distribution<double>(0.0, total)(rng);
    for (size_t i = 0; i < w.size(); ++i) {
        r -= w[i];
        if (r <= 0) return members_[i].id;
    }
    return members_.back().id;
}

void OpponentPool::loadInto(int id, PolicyNet& replica) const {
    const Snapshot* s = nullptr;
    for (const auto& m : members_)
        if (m.id == id) s = &m;
    TORCH_CHECK(s != nullptr, "pool member not found: ", id);
    torch::NoGradGuard ng;
    auto dp = replica->parameters();
    auto db = replica->buffers();
    TORCH_CHECK(dp.size() + db.size() == s->params.size(), "pool snapshot shape mismatch");
    size_t k = 0;
    for (auto& p : dp) p.copy_(s->params[k++]);
    for (auto& b : db) b.copy_(s->params[k++]);
}

void OpponentPool::recordResult(int id, double learnerScore) {
    Snapshot* s = find(id);
    if (!s) return;
    s->winEma = (1.0 - emaAlpha_) * s->winEma + emaAlpha_ * learnerScore;
    ++s->games;
}

void OpponentPool::saveMeta(const std::string& dir) const {
    fs::create_directories(dir + "/pool");
    std::ofstream f(dir + "/pool/meta.csv", std::ios::trunc);
    f << "id,born_update,elo,win_ema,games\n";
    for (const auto& s : members_)
        f << s.id << "," << s.bornUpdate << "," << s.elo << "," << s.winEma << "," << s.games << "\n";
    f << "# next_id," << nextId_ << "\n";
}

void OpponentPool::load(const std::string& dir) {
    members_.clear();
    std::ifstream f(dir + "/pool/meta.csv");
    if (!f) return;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            auto pos = line.find(',');
            if (pos != std::string::npos) nextId_ = std::stoi(line.substr(pos + 1));
            continue;
        }
        Snapshot s;
        std::stringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ','); s.id = std::stoi(tok);
        std::getline(ss, tok, ','); s.bornUpdate = std::stol(tok);
        std::getline(ss, tok, ','); s.elo = std::stod(tok);
        std::getline(ss, tok, ','); s.winEma = std::stod(tok);
        std::getline(ss, tok, ','); s.games = std::stol(tok);
        std::string p = snapPath(dir, s.id);
        if (!fs::exists(p)) continue;
        torch::load(s.params, p);
        members_.push_back(std::move(s));
    }
}

}
