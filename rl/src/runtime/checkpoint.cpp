#include "runtime/checkpoint.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace rl {

void TrainState::save(const std::string& path) const {
    std::ofstream f(path, std::ios::trunc);
    f << "update=" << update << "\n";
    f << "env_steps=" << envSteps << "\n";
    f << "elo=" << elo << "\n";
    for (const auto& [k, v] : extra) f << k << "=" << v << "\n";
}

bool TrainState::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string k = line.substr(0, pos);
        double v = std::stod(line.substr(pos + 1));
        if (k == "update") update = (long)v;
        else if (k == "env_steps") envSteps = (long)v;
        else if (k == "elo") elo = v;
        else extra[k] = v;
    }
    return true;
}

void saveCheckpoint(const std::string& dir, PolicyNet& policy, CriticNet& critic,
                    torch::optim::Adam& opt, const TrainState& st) {
    fs::create_directories(dir);
    torch::save(policy, dir + "/policy.pt.tmp");
    fs::rename(dir + "/policy.pt.tmp", dir + "/policy.pt");
    if (!critic.is_empty()) {
        torch::save(critic, dir + "/critic.pt.tmp");
        fs::rename(dir + "/critic.pt.tmp", dir + "/critic.pt");
    }
    torch::save(opt, dir + "/optim.pt.tmp");
    fs::rename(dir + "/optim.pt.tmp", dir + "/optim.pt");
    st.save(dir + "/state.txt");
}

bool loadCheckpoint(const std::string& dir, PolicyNet& policy, CriticNet& critic,
                    torch::optim::Adam& opt, TrainState& st, torch::Device device) {
    if (!fs::exists(dir + "/policy.pt") || !fs::exists(dir + "/state.txt")) return false;
    torch::load(policy, dir + "/policy.pt", device);
    if (!critic.is_empty() && fs::exists(dir + "/critic.pt"))
        torch::load(critic, dir + "/critic.pt", device);
    if (fs::exists(dir + "/optim.pt")) torch::load(opt, dir + "/optim.pt", device);
    return st.load(dir + "/state.txt");
}

bool loadPolicyWeights(const std::string& dir, PolicyNet& policy) {
    std::string p = fs::is_directory(dir) ? dir + "/policy.pt" : dir;
    if (!fs::exists(p)) return false;
    torch::load(policy, p, torch::Device(torch::kCPU));
    return true;
}

bool saveBestCheckpoint(const std::string& dir, PolicyNet& policy, CriticNet& critic,
                        double elo, long update) {
    double bestElo = -1e18;
    {
        std::ifstream f(dir + "/best.txt");
        std::string line;
        while (f && std::getline(f, line)) {
            auto pos = line.find('=');
            if (pos != std::string::npos && line.substr(0, pos) == "elo")
                bestElo = std::stod(line.substr(pos + 1));
        }
    }
    if (elo <= bestElo) return false;

    fs::create_directories(dir);
    torch::save(policy, dir + "/policy_best.pt.tmp");
    fs::rename(dir + "/policy_best.pt.tmp", dir + "/policy_best.pt");
    if (!critic.is_empty()) {
        torch::save(critic, dir + "/critic_best.pt.tmp");
        fs::rename(dir + "/critic_best.pt.tmp", dir + "/critic_best.pt");
    }
    std::ofstream f(dir + "/best.txt", std::ios::trunc);
    f << "elo=" << elo << "\n";
    f << "update=" << update << "\n";
    return true;
}

}
