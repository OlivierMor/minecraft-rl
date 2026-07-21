#include "vec/vec_env.h"

#include <algorithm>
#include <stdexcept>

#include "env/registry.h"

namespace rl {

VecEnv::VecEnv(const Config& cfg, int numEnvs, uint64_t seed, WorkerPool& pool)
    : workers_(pool) {
    envs_.reserve((size_t)numEnvs);
    for (int i = 0; i < numEnvs; ++i)
        envs_.push_back(std::make_unique<Env>(i, seed + (uint64_t)i * 9973ULL,
                                              registry::makeComponents(cfg)));
    bots_.resize((size_t)numEnvs);
    float turn = (float)cfg.num("bot.max_turn", 40.0);
    int click = (int)cfg.num("bot.click_interval", 3);
    for (auto& b : bots_) {
        b.maxTurnPerTick = turn;
        b.clickInterval = click;
    }
    obsDim_ = envs_[0]->obsSize();
    criticObsDim_ = envs_[0]->criticObsSize();
    spec_ = envs_[0]->spec();
    workerStats_.resize((size_t)workers_.size());
    setPartition(numEnvs, 0, 0, 0);
}

void VecEnv::setPartition(int mirror, int poolEnvs, int scripted, int poolGroups) {
    if (mirror + poolEnvs + scripted != N())
        throw std::runtime_error("VecEnv::setPartition: counts must sum to N");
    if (poolEnvs > 0 && poolGroups < 1)
        throw std::runtime_error("VecEnv::setPartition: pool envs need >= 1 group");
    mirror_ = mirror;
    poolEnvs_ = poolEnvs;
    scripted_ = scripted;
    groupStart_.clear();
    groupEnd_.clear();
    groupOfEnv_.assign((size_t)N(), -1);
    if (poolEnvs > 0) {
        for (int g = 0; g < poolGroups; ++g) {
            auto [b, e] = WorkerPool::slice(poolEnvs, g, poolGroups);
            groupStart_.push_back(mirror + b);
            groupEnd_.push_back(mirror + e);
            for (int i = mirror + b; i < mirror + e; ++i) groupOfEnv_[(size_t)i] = (int16_t)g;
        }
    }
}

void VecEnv::buildObs(float* learnerOut, float* oppOut, float* criticOut,
                      int e0, int e1) {
    const int n = N(), sideRows = mirror_ + poolEnvs_;
    if (e1 < 0) e1 = n;
    std::atomic<int> cursor{e0};
    constexpr int CHUNK = 32;
    auto job = [&](int) {
        for (int b; (b = cursor.fetch_add(CHUNK, std::memory_order_relaxed)) < e1;) {
        const int hi = std::min(e1, b + CHUNK);
        for (int e = b; e < hi; ++e) {
            Env& env = *envs_[(size_t)e];
            env.buildObs(0, learnerOut + (size_t)e * obsDim_);
            if (e < mirror_)
                env.buildObs(1, learnerOut + ((size_t)n + e) * obsDim_);
            else if (e < sideRows)
                env.buildObs(1, oppOut + (size_t)(e - mirror_) * obsDim_);
            if (criticOut) {
                env.buildCriticObs(0, criticOut + (size_t)e * criticObsDim_);
                if (e < mirror_)
                    env.buildCriticObs(1, criticOut + ((size_t)n + e) * criticObsDim_);
            }
        }
        }
    };
    workers_.run(job);
}

void VecEnv::stepRange(int e0, int e1, const int32_t* actions, float* rewardOut,
                       uint8_t* doneOut, TruncSink* trunc, int tStep, int worker) {
    const int n = N(), nb = spec_.numBranches(), sideRows = mirror_ + poolEnvs_;
    for (int e = e0; e < e1; ++e) {
        Env& env = *envs_[(size_t)e];
        mc::Input in0 = env.parseAction(0, actions + (size_t)e * nb);
        mc::Input in1;
        if (e < sideRows)
            in1 = env.parseAction(1, actions + ((size_t)n + e) * nb);
        else
            in1 = bots_[(size_t)e].act(env.sim.client[1], env.sim.tickCount);

        Env::StepOut out = env.step(in0, in1);

        rewardOut[e] = out.reward[0];
        doneOut[e] = out.done ? 1 : 0;
        if (e < mirror_) {
            rewardOut[n + e] = out.reward[1];
            doneOut[n + e] = out.done ? 1 : 0;
        }

        if (out.done) {
            if (out.truncated && trunc) {
                int learners = e < mirror_ ? 2 : 1;
                for (int s = 0; s < learners; ++s) {
                    int slot = trunc->n.fetch_add(1, std::memory_order_relaxed);
                    if (slot >= trunc->cap) {
                        trunc->dropped.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    int agentRow = s == 0 ? e : n + e;
                    env.buildObs(s, trunc->obs + (size_t)slot * obsDim_);
                    if (trunc->cobs)
                        env.buildCriticObs(s, trunc->cobs + (size_t)slot * criticObsDim_);
                    trunc->agent[slot] = agentRow;
                    trunc->t[slot] = tStep;
                }
            }
            EpisodeStat st;
            st.env = e;
            st.oppKind = e < mirror_ ? 0 : (e < sideRows ? 1 : 2);
            st.oppGroup = groupOfEnv_[(size_t)e];
            st.winner = (int8_t)out.winner;
            st.truncated = out.truncated;
            st.ticks = env.sim.tickCount;
            st.hits[0] = env.sim.sv[0].hits;
            st.hits[1] = env.sim.sv[1].hits;
            for (int s = 0; s < 2; ++s) {
                st.crits[s] = env.fight.crits[s];
                st.jumps[s] = env.fight.jumps[s];
                st.maxCombo[s] = env.fight.maxCombo[s];
            }
            workerStats_[(size_t)worker].push_back(st);
            env.reset();
        }
    }
}

void VecEnv::step(const int32_t* actions, float* rewardOut, uint8_t* doneOut,
                  TruncSink* trunc, int tStep, int e0, int e1) {
    if (e1 < 0) e1 = N();
    std::atomic<int> cursor{e0};
    constexpr int CHUNK = 32;
    auto job = [&](int w) {
        for (int b; (b = cursor.fetch_add(CHUNK, std::memory_order_relaxed)) < e1;)
            stepRange(b, std::min(e1, b + CHUNK), actions, rewardOut, doneOut, trunc, tStep, w);
    };
    workers_.run(job);
}

void VecEnv::stepSerial(const int32_t* actions, float* rewardOut, uint8_t* doneOut,
                        TruncSink* trunc, int tStep) {
    stepRange(0, N(), actions, rewardOut, doneOut, trunc, tStep, 0);
}

std::vector<EpisodeStat> VecEnv::drainStats() {
    std::vector<EpisodeStat> out;
    for (auto& v : workerStats_) {
        out.insert(out.end(), v.begin(), v.end());
        v.clear();
    }
    return out;
}

std::vector<std::pair<std::string, double>> VecEnv::rewardStats() {
    std::vector<std::pair<std::string, double>> merged;
    std::vector<std::pair<std::string, double>> one;
    for (auto& env : envs_) {
        one.clear();
        env->rewardFn().collectStats(one);
        if (merged.empty()) {
            merged = one;
        } else {
            for (size_t i = 0; i < one.size() && i < merged.size(); ++i)
                merged[i].second += one[i].second;
        }
    }
    for (auto& [k, v] : merged) v /= (double)envs_.size();
    return merged;
}

}
