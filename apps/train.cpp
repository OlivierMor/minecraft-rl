#include <torch/torch.h>

#if __has_include(<c10/cuda/CUDAStream.h>) && __has_include(<c10/cuda/impl/cuda_cmake_macros.h>)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#define MCRL_HAS_CUDA_STREAMS 1
#else
#define MCRL_HAS_CUDA_STREAMS 0
#endif
#if MCRL_HAS_CUDA_STREAMS && __has_include(<ATen/cuda/CUDAGraph.h>)
#include <ATen/cuda/CUDAGraph.h>
#define MCRL_HAS_CUDA_GRAPHS 1
#else
#define MCRL_HAS_CUDA_GRAPHS 0
#endif
#if __has_include(<ATen/autocast_mode.h>)
#include <ATen/autocast_mode.h>
#define MCRL_HAS_AUTOCAST 1
#else
#define MCRL_HAS_AUTOCAST 0
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <thread>

#include "config.h"
#include "model/distributions.h"
#include "model/policy.h"
#include "ppo/learner.h"
#include "ppo/rollout.h"
#include "runtime/checkpoint.h"
#include "runtime/logger.h"
#include "selfplay/elo.h"
#include "selfplay/match.h"
#include "selfplay/pool.h"
#include "vec/vec_env.h"

namespace fs = std::filesystem;
using namespace rl;
using Clock = std::chrono::steady_clock;

static volatile std::sig_atomic_t g_stop = 0;
static void onSigint(int) { g_stop = 1; }

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) try {
    std::string cfgPath = argc > 1 ? argv[1] : MC_ENV_DEFAULT_CONFIG;
    bool resume = false;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--resume") resume = true;

    Config cfg = Config::fromFile(cfgPath);

    std::string devStr = cfg.str("run.device", "auto");
    torch::Device device = torch::kCPU;
    if (devStr == "auto")
        device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    else
        device = torch::Device(devStr);
    uint64_t seed = (uint64_t)cfg.num("run.seed", 1);
    torch::manual_seed(seed);
    int hw = (int)std::thread::hardware_concurrency();
    int workers = (int)cfg.num("run.workers", std::max(1, hw - 2));
    torch::set_num_threads((int)cfg.num("run.torch_threads", device.is_cuda() ? 2 : std::max(1, hw / 2)));
    if (device.is_cuda()) {
        at::globalContext().setAllowTF32CuBLAS(true);
        at::globalContext().setAllowTF32CuDNN(true);
    }

    std::string runName = cfg.str("run.name", "run");
    std::string dir = cfg.str("run.checkpoint_dir", "runs") + "/" + runName;
    fs::create_directories(dir + "/replays");
    if (!resume) {
        std::ofstream(dir + "/config.toml") << cfg.rawText();
    }

    WorkerPool pool(workers);
    int N = (int)cfg.num("env.num_envs", 4096);
    VecEnv vec(cfg, N, seed, pool);
    const int Do = vec.obsDim(), Dc = vec.criticObsDim();
    const ActionSpec spec = vec.spec();
    const int NB = spec.numBranches();
    const int M = std::min(N, std::max(0, (int)std::lround(N * cfg.num("selfplay.mirror_fraction", 0.5))));
    const int A = N + M;
    const int T = (int)cfg.num("ppo.horizon", 128);

    PolicyNet policy = buildPolicy(cfg, Do, spec);
    CriticNet critic = Dc > 0 ? buildCritic(cfg, Dc) : CriticNet(nullptr);
    Learner learner(policy, critic, device, cfg);
    PPOParams pp = PPOParams::fromConfig(cfg);

    OpponentPool oppPool(cfg);
    EloTracker elo(cfg.num("elo.anchor", 1000), cfg.num("elo.k_current", 8), cfg.num("elo.k_pool", 4));
    TrainState st;
    if (resume) {
        if (!loadCheckpoint(dir, policy, critic, learner.optimizer(), st, device))
            throw std::runtime_error("--resume: no checkpoint in " + dir);
        oppPool.load(dir);
        elo.setCurrent(st.elo);
        std::printf("resumed %s at update %ld (%.0f elo, pool %d)\n", dir.c_str(), st.update,
                    st.elo, oppPool.size());
    }
    elo.openMatchLog(dir + "/matches.csv");

    const int G = std::max(1, (int)cfg.num("selfplay.max_active_opponents", 4));
    std::vector<PolicyNet> replicas;
    std::vector<int> groupIds;
    for (int g = 0; g < G; ++g) {
        PolicyNet r = buildPolicy(cfg, Do, spec);
        r->to(device);
        r->eval();
        replicas.push_back(r);
    }
    PolicyNet evalReplica = buildPolicy(cfg, Do, spec);
    evalReplica->to(device);
    evalReplica->eval();
    std::mt19937_64 rng(seed ^ 0x5be1fabull);

    MatchRunner matches(cfg, (int)cfg.num("eval.parallel", 64), seed, pool, device);

    auto hostOpts = torch::TensorOptions().dtype(torch::kFloat32);
    auto hostI32 = torch::TensorOptions().dtype(torch::kInt32);
    if (device.is_cuda()) {
        hostOpts = hostOpts.pinned_memory(true);
        hostI32 = hostI32.pinned_memory(true);
    }
    torch::Tensor oppObsHost = torch::zeros({N, Do}, hostOpts);
    torch::Tensor actionsHost = torch::zeros({2L * N, NB}, hostI32);
    torch::Tensor finalObsHost = torch::zeros({A, Do}, hostOpts);
    torch::Tensor finalCobsHost = Dc > 0 ? torch::zeros({A, Dc}, hostOpts) : torch::Tensor();
    float* oppObsPtr = oppObsHost.data_ptr<float>();
    const bool useOverlap = cfg.boolean("collect.overlap", true) && M > 0 && M < N;
    bool useGraphs = cfg.boolean("collect.cuda_graph", true) && device.is_cuda();
    (void)useGraphs;
    const bool collectBf16 =
        cfg.boolean("collect.bf16", true) && device.is_cuda() && MCRL_HAS_AUTOCAST;

    CsvLogger log(dir + "/train_log.csv");
    std::signal(SIGINT, onSigint);
    std::printf("training '%s' on %s | envs %d (mirror %d) agents %d obs %d critic %d "
                "branches %d horizon %d workers %d\n",
                runName.c_str(), device.str().c_str(), N, M, A, Do, Dc, NB, T, workers);

    const long totalUpdates = (long)cfg.num("run.total_updates", 100000);
    const int resampleEvery = (int)cfg.num("selfplay.resample_every", 8);
    const int poolInterval = (int)cfg.num("selfplay.pool_interval", 200);
    const int evalInterval = (int)cfg.num("eval.interval", 100);
    const int ckptInterval = (int)cfg.num("run.checkpoint_interval", 100);
    const bool oppDeterministic = cfg.boolean("selfplay.opponent_deterministic", false);
    const bool evalDeterministic = cfg.boolean("eval.deterministic", true);
    const int evalMatches = (int)cfg.num("eval.matches", 128);
    const int evalPoolOpponents = (int)cfg.num("eval.pool_opponents", 3);
    const bool dumpReplays = cfg.boolean("log.replay_dump", true);
    const long logInterval = (long)cfg.num("log.interval", 10);
    const bool lrAnneal = cfg.boolean("ppo.lr_anneal", true);
    cfg.schedule("selfplay.scripted_fraction_schedule", 0.0,
                 cfg.num("selfplay.scripted_fraction", 0.1));
    cfg.schedule("ppo.entropy_schedule", 0.0, pp.entCoef);
    for (const auto& k : cfg.unusedKeys()) {
        if (k.rfind("viewer.", 0) == 0) continue;
        std::printf("WARNING: config key never read (typo?): %s\n", k.c_str());
    }

    PolicyNet actorPolicy = buildPolicy(cfg, Do, spec);
    actorPolicy->to(device);
    actorPolicy->eval();
    CriticNet actorCritic = Dc > 0 ? buildCritic(cfg, Dc) : CriticNet(nullptr);
    if (Dc > 0) {
        actorCritic->to(device);
        actorCritic->eval();
    }
    auto syncActors = [&]() {
        copyWeights(*policy, *actorPolicy);
        if (Dc > 0) copyWeights(*critic, *actorCritic);
        if (device.is_cuda()) torch::cuda::synchronize();
    };

    struct Chunk { int row0 = 0, len = 0; };
    struct Half {
        int e0 = 0, e1 = 0;
        std::vector<Chunk> chunks;
        int rows = 0;
        int oppRows = 0;
        std::vector<int> groups;
        torch::Tensor obsDev, cobsDev;
        torch::Tensor actsDev, oppActsDev;
        torch::Tensor lpvalDev;
        torch::Tensor actsHost, oppActsHost, lpvalHost;
#if MCRL_HAS_CUDA_GRAPHS
        std::unique_ptr<at::cuda::CUDAGraph> graph;
        std::string sig;
#endif
    };
    auto devF32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto devI32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    torch::Tensor oppObsDev = torch::zeros({std::max(1, N - M), Do}, devF32);
    auto makeHalf = [&](int e0, int e1, std::vector<Chunk> chunks) {
        Half h;
        h.e0 = e0;
        h.e1 = e1;
        h.chunks = std::move(chunks);
        for (const Chunk& c : h.chunks) h.rows += c.len;
        h.obsDev = torch::zeros({h.rows, Do}, devF32);
        if (Dc > 0) h.cobsDev = torch::zeros({h.rows, Dc}, devF32);
        h.actsDev = torch::zeros({h.rows, NB}, devI32);
        h.oppActsDev = torch::zeros({std::max(1, N - M), NB}, devI32);
        h.lpvalDev = torch::zeros({2L * h.rows}, devF32);
        h.actsHost = torch::zeros({h.rows, NB}, hostI32);
        h.oppActsHost = torch::zeros({std::max(1, N - M), NB}, hostI32);
        h.lpvalHost = torch::zeros({2L * h.rows}, hostOpts);
        return h;
    };
    std::vector<Half> halves;
    if (useOverlap) {
        halves.push_back(makeHalf(0, M, {{0, M}, {N, M}}));
        halves.push_back(makeHalf(M, N, {{M, N - M}}));
    } else {
        halves.push_back(makeHalf(0, N, {{0, A}}));
    }
    auto refreshHalves = [&]() {
        for (Half& h : halves) {
            h.groups.clear();
            h.oppRows = 0;
        }
        Half& oppHalf = halves.back();
        oppHalf.oppRows = vec.P();
        for (int g = 0; g < vec.poolGroups(); ++g) {
            auto [ob, oe] = vec.poolGroupOppRows(g);
            if (oe > ob) oppHalf.groups.push_back(g);
        }
    };

    auto actorMath = [&](Half& h) {
        torch::Tensor logits, value;
        torch::Tensor oppLogits[8];
#if MCRL_HAS_AUTOCAST
        if (collectBf16) {
            at::autocast::set_autocast_enabled(at::kCUDA, true);
            at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
            at::autocast::set_autocast_cache_enabled(false);
        }
#endif
        if (Dc > 0) {
            logits = actorPolicy->logitsOnly(h.obsDev);
            value = actorCritic->forward(h.cobsDev);
        } else {
            std::tie(logits, value) = actorPolicy->forward(h.obsDev);
        }
        for (size_t k = 0; k < h.groups.size() && k < 8; ++k) {
            auto [ob, oe] = vec.poolGroupOppRows(h.groups[k]);
            oppLogits[k] =
                replicas[(size_t)h.groups[k]]->logitsOnly(oppObsDev.narrow(0, ob, oe - ob));
        }
#if MCRL_HAS_AUTOCAST
        if (collectBf16) at::autocast::set_autocast_enabled(at::kCUDA, false);
#endif
        auto [acts, lp] = sampleWithLogProb(logits.to(torch::kFloat32), spec, false);
        h.actsDev.copy_(acts.to(torch::kInt32));
        h.lpvalDev.narrow(0, 0, h.rows).copy_(lp);
        h.lpvalDev.narrow(0, h.rows, h.rows).copy_(value.to(torch::kFloat32));
        for (size_t k = 0; k < h.groups.size() && k < 8; ++k) {
            auto [ob, oe] = vec.poolGroupOppRows(h.groups[k]);
            torch::Tensor oa =
                sampleActions(oppLogits[k].to(torch::kFloat32), spec, oppDeterministic);
            h.oppActsDev.narrow(0, ob, oe - ob).copy_(oa.to(torch::kInt32));
        }
    };

    auto launchHalf = [&](Half& h, Rollout& roll, int t) {
        torch::Tensor obsT = roll.obs[t];
        torch::Tensor cobsT = Dc > 0 ? roll.cobs[t] : torch::Tensor();
        int dst = 0;
        for (const Chunk& c : h.chunks) {
            h.obsDev.narrow(0, dst, c.len).copy_(obsT.narrow(0, c.row0, c.len), true);
            if (Dc > 0)
                h.cobsDev.narrow(0, dst, c.len).copy_(cobsT.narrow(0, c.row0, c.len), true);
            dst += c.len;
        }
        if (h.oppRows > 0)
            oppObsDev.narrow(0, 0, h.oppRows).copy_(oppObsHost.narrow(0, 0, h.oppRows), true);
#if MCRL_HAS_CUDA_GRAPHS
        if (useGraphs) {
            std::string sig = std::to_string(h.oppRows);
            for (int g : h.groups) {
                auto [ob, oe] = vec.poolGroupOppRows(g);
                sig += "|" + std::to_string(ob) + ":" + std::to_string(oe);
            }
            if (!h.graph || h.sig != sig) {
                h.graph.reset();
                h.sig.clear();
                try {
                    actorMath(h);
                    actorMath(h);
                    c10::cuda::getCurrentCUDAStream().synchronize();
                    auto g = std::make_unique<at::cuda::CUDAGraph>();
                    bool capturing = false;
                    try {
                        g->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
                        capturing = true;
                        actorMath(h);
                        g->capture_end();
                        capturing = false;
                    } catch (...) {
                        if (capturing) {
                            try { g->capture_end(); } catch (const std::exception&) {}
                        }
                        throw;
                    }
                    c10::cuda::getCurrentCUDAStream().synchronize();
                    h.graph = std::move(g);
                    h.sig = sig;
                } catch (const std::exception& e) {
                    std::fprintf(stderr,
                                 "collector: CUDA graph capture failed (%s) - eager fallback\n",
                                 e.what());
                    useGraphs = false;
                }
            }
            if (useGraphs && h.graph && h.sig == sig) h.graph->replay();
            else actorMath(h);
        } else {
            actorMath(h);
        }
#else
        actorMath(h);
#endif
        h.actsHost.copy_(h.actsDev, true);
        h.lpvalHost.copy_(h.lpvalDev, true);
        if (h.oppRows > 0)
            h.oppActsHost.narrow(0, 0, h.oppRows)
                .copy_(h.oppActsDev.narrow(0, 0, h.oppRows), true);
    };

    auto syncStream = [&]() {
#if MCRL_HAS_CUDA_STREAMS
        if (device.is_cuda()) c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
#endif
    };

    auto finishHalf = [&](Half& h, Rollout& roll, int t) {
        syncStream();
        const int32_t* aSrc = h.actsHost.data_ptr<int32_t>();
        const float* lv = h.lpvalHost.data_ptr<float>();
        int32_t* aAll = actionsHost.data_ptr<int32_t>();
        int32_t* aRoll = roll.actions.data_ptr<int32_t>() + (size_t)t * A * NB;
        float* lpDst = roll.logp.data_ptr<float>() + (size_t)t * A;
        float* vDst = roll.value.data_ptr<float>() + (size_t)t * A;
        int dst = 0;
        for (const Chunk& c : h.chunks) {
            std::memcpy(aAll + (size_t)c.row0 * NB, aSrc + (size_t)dst * NB,
                        (size_t)c.len * NB * sizeof(int32_t));
            std::memcpy(aRoll + (size_t)c.row0 * NB, aSrc + (size_t)dst * NB,
                        (size_t)c.len * NB * sizeof(int32_t));
            std::memcpy(lpDst + c.row0, lv + dst, (size_t)c.len * sizeof(float));
            std::memcpy(vDst + c.row0, lv + h.rows + dst, (size_t)c.len * sizeof(float));
            dst += c.len;
        }
        if (h.oppRows > 0)
            std::memcpy(aAll + (size_t)A * NB, h.oppActsHost.data_ptr<int32_t>(),
                        (size_t)h.oppRows * NB * sizeof(int32_t));
    };

    struct Batch {
        Rollout roll;
        torch::Tensor adv, ret;
    };
    Batch bufA{Rollout(T, A, Do, Dc, NB, device.is_cuda()), {}, {}};
    Batch bufB{Rollout(T, A, Do, Dc, NB, device.is_cuda()), {}, {}};
    Batch* trainB = &bufA;
    Batch* collectB = &bufB;

    std::atomic<double> lastCollectMs{0.0};

    auto collectBatch = [&](Batch& b, long batchIdx) {
        auto c0 = Clock::now();
        torch::NoGradGuard ng;
#if MCRL_HAS_CUDA_STREAMS
        std::optional<c10::cuda::CUDAStreamGuard> streamGuard;
        if (device.is_cuda())
            streamGuard.emplace(
                c10::cuda::getStreamFromPool(true, device.index()));
#endif
        vec.onUpdate(batchIdx);
        Rollout& roll = b.roll;
        roll.beginRollout();
        const int32_t* actPtr = actionsHost.data_ptr<int32_t>();
        float* rewBase = roll.reward.data_ptr<float>();
        uint8_t* doneBase = roll.done.data_ptr<uint8_t>();
        auto obsPtrAt = [&](int t) { return roll.obs.data_ptr<float>() + (size_t)t * A * Do; };
        auto cobsPtrAt = [&](int t) {
            return Dc > 0 ? roll.cobs.data_ptr<float>() + (size_t)t * A * Dc : nullptr;
        };
        if (useOverlap) {
            Half& h0 = halves[0];
            Half& h1 = halves[1];
            vec.buildObs(obsPtrAt(0), oppObsPtr, cobsPtrAt(0), h0.e0, h0.e1);
            for (int t = 0; t < T; ++t) {
                launchHalf(h0, roll, t);
                if (t > 0)
                    vec.step(actPtr, rewBase + (size_t)(t - 1) * A,
                             doneBase + (size_t)(t - 1) * A, &roll.sink, t - 1, h1.e0, h1.e1);
                vec.buildObs(obsPtrAt(t), oppObsPtr, cobsPtrAt(t), h1.e0, h1.e1);
                finishHalf(h0, roll, t);
                launchHalf(h1, roll, t);
                vec.step(actPtr, rewBase + (size_t)t * A,
                         doneBase + (size_t)t * A, &roll.sink, t, h0.e0, h0.e1);
                if (t + 1 < T)
                    vec.buildObs(obsPtrAt(t + 1), oppObsPtr, cobsPtrAt(t + 1), h0.e0, h0.e1);
                finishHalf(h1, roll, t);
            }
            vec.step(actPtr, rewBase + (size_t)(T - 1) * A, doneBase + (size_t)(T - 1) * A,
                     &roll.sink, T - 1, h1.e0, h1.e1);
        } else {
            Half& h = halves[0];
            vec.buildObs(obsPtrAt(0), oppObsPtr, cobsPtrAt(0));
            for (int t = 0; t < T; ++t) {
                launchHalf(h, roll, t);
                finishHalf(h, roll, t);
                vec.step(actPtr, rewBase + (size_t)t * A, doneBase + (size_t)t * A,
                         &roll.sink, t);
                if (t + 1 < T) vec.buildObs(obsPtrAt(t + 1), oppObsPtr, cobsPtrAt(t + 1));
            }
        }

        vec.buildObs(finalObsHost.data_ptr<float>(), oppObsPtr,
                     Dc > 0 ? finalCobsHost.data_ptr<float>() : nullptr);
        torch::Tensor lastValue, truncValues = torch::zeros({0}, torch::kFloat32);
        if (Dc > 0) {
            lastValue = actorCritic->forward(finalCobsHost.to(device, true)).to(torch::kCPU);
        } else {
            torch::Tensor learnerObs = finalObsHost.to(device, true);
            lastValue = actorPolicy->forward(learnerObs).second.to(torch::kCPU);
        }
        int nTrunc = std::min(roll.sink.n.load(), roll.sink.cap);
        if (nTrunc > 0) {
            if (Dc > 0) {
                torch::Tensor tCobs = roll.truncCobs.narrow(0, 0, nTrunc).to(device, true);
                truncValues = actorCritic->forward(tCobs).to(torch::kCPU);
            } else {
                torch::Tensor tObs = roll.truncObs.narrow(0, 0, nTrunc).to(device, true);
                truncValues = actorPolicy->forward(tObs).second.to(torch::kCPU);
            }
        }
        if (roll.sink.dropped.load() > 0)
            std::fprintf(stderr, "WARNING: %d truncation bootstraps dropped (sink overflow)\n",
                         roll.sink.dropped.load());
        std::tie(b.adv, b.ret) = roll.computeGAE(lastValue, truncValues, pp.gamma, pp.lam);
        lastCollectMs.store(ms(c0, Clock::now()));
    };

    std::thread collector;

    double winVsScripted = 0.5, winVsPool = 0.5;
    long epCount = 0;
    double epLenSum = 0, epHitsSum = 0;
    double epCritsSum = 0, epJumpsSum = 0, epMaxComboSum = 0;

    auto resampleFor = [&](long batchIdx) {
        if (batchIdx % resampleEvery != 0 && !groupIds.empty()) return;
        double sFrac = cfg.schedule("selfplay.scripted_fraction_schedule", (double)batchIdx,
                                    cfg.num("selfplay.scripted_fraction", 0.1));
        int S = std::min(N - M, std::max(0, (int)std::lround(N * sFrac)));
        int q = std::max(1, N / 16);
        S = std::min(N - M, (int)((S + q / 2) / q) * q);
        int P = N - M - S;
        int gActual = P > 0 ? std::min(G, std::max(1, P)) : 0;
        groupIds.assign((size_t)std::max(gActual, 1), -1);
        if (P > 0) {
            for (int g = 0; g < gActual; ++g) {
                int id = oppPool.empty() ? -1 : oppPool.samplePFSP(rng);
                groupIds[(size_t)g] = id;
                if (id >= 0) oppPool.loadInto(id, replicas[(size_t)g]);
                else copyWeights(*policy, *replicas[(size_t)g]);
            }
        }
        vec.setPartition(M, P, S, gActual);
        refreshHalves();
    };

    resampleFor(st.update);
    syncActors();
    collectBatch(*collectB, st.update);
    std::swap(trainB, collectB);

    for (long update = st.update; update < totalUpdates && !g_stop; ++update) {
        auto t0 = Clock::now();

        resampleFor(update + 1);
        syncActors();
        collector = std::thread([&collectBatch, collectB, update] {
            collectBatch(*collectB, update + 1);
        });
        auto t1 = Clock::now();

        policy->train();
        double frac = 1.0 - (double)update / (double)totalUpdates;
        PPOParams p = pp;
        if (lrAnneal) p.lr = (float)(pp.lr * std::max(0.05, frac));
        p.entCoef = (float)cfg.schedule("ppo.entropy_schedule", (double)update, pp.entCoef);
        PPOStats ps = learner.update(trainB->roll, trainB->adv, trainB->ret, p);
        auto t2 = Clock::now();

        collector.join();
        std::swap(trainB, collectB);
        auto t3 = Clock::now();

        for (const EpisodeStat& e : vec.drainStats()) {
            ++epCount;
            epLenSum += e.ticks;
            epHitsSum += e.hits[0] + e.hits[1];
            epCritsSum += e.crits[0];
            epJumpsSum += e.jumps[0];
            epMaxComboSum += e.maxCombo[0];
            double score = e.winner < 0 ? 0.5 : (e.winner == 0 ? 1.0 : 0.0);
            if (e.oppKind == 2) winVsScripted = 0.98 * winVsScripted + 0.02 * score;
            if (e.oppKind == 1 && e.oppGroup >= 0 && e.oppGroup < (int)groupIds.size()) {
                winVsPool = 0.98 * winVsPool + 0.02 * score;
                int id = groupIds[(size_t)e.oppGroup];
                if (id >= 0) oppPool.recordResult(id, score);
            }
        }

        st.update = update + 1;
        st.envSteps += (long)T * A;

        if (poolInterval > 0 && st.update % poolInterval == 0)
            oppPool.snapshot(policy, elo.current(), st.update, dir);

        if (evalInterval > 0 && st.update % evalInterval == 0) {
            policy->eval();
            MatchController cur{policy, evalDeterministic};
            MatchController bot{PolicyNet(nullptr), true};
            Replay rep;
            MatchOutcome vsBot = matches.run(cur, bot, evalMatches, dumpReplays ? &rep : nullptr);
            for (int i = 0; i < vsBot.wins; ++i) elo.applyResult(elo.anchor(), true, 1.0);
            for (int i = 0; i < vsBot.draws; ++i) elo.applyResult(elo.anchor(), true, 0.5);
            for (int i = 0; i < vsBot.losses; ++i) elo.applyResult(elo.anchor(), true, 0.0);
            elo.logMatch(st.update, "bot", elo.anchor(), vsBot.score());
            if (dumpReplays)
                rep.save(dir + "/replays/eval_" + std::to_string(st.update) + "_vs_bot.mcrp");

            int nOpp = std::min(evalPoolOpponents, oppPool.size());
            for (int k = 0; k < nOpp; ++k) {
                int id = oppPool.samplePFSP(rng);
                Snapshot* snap = oppPool.find(id);
                if (!snap) continue;
                oppPool.loadInto(id, evalReplica);
                MatchController oppc{evalReplica, evalDeterministic};
                MatchOutcome o = matches.run(cur, oppc, std::max(16, evalMatches / 4));
                double newOpp = elo.applyResult(snap->elo, false, o.score());
                snap->elo = newOpp;
                oppPool.recordResult(id, o.score());
                elo.logMatch(st.update, "snap_" + std::to_string(id), snap->elo, o.score());
            }
            if (saveBestCheckpoint(dir, policy, critic, elo.current(), st.update))
                std::printf("  new best elo %.0f @%ld -> policy_best.pt\n", elo.current(),
                            st.update);
            std::printf("  eval @%ld: vs bot %d-%d-%d (%.2f) | elo %.0f | pool %d\n", st.update,
                        vsBot.wins, vsBot.draws, vsBot.losses, vsBot.score(), elo.current(),
                        oppPool.size());
        }

        st.elo = elo.current();
        if (ckptInterval > 0 && st.update % ckptInterval == 0) {
            saveCheckpoint(dir, policy, critic, learner.optimizer(), st);
            oppPool.saveMeta(dir);
        }

        auto t4 = Clock::now();
        double collectMs = lastCollectMs.load();
        double launchMs = ms(t0, t1), updateMs = ms(t1, t2), stallMs = ms(t2, t3);
        double otherMs = ms(t3, t4) + launchMs;
        double sps = (double)T * A / (ms(t0, t4) / 1000.0);
        std::vector<std::pair<std::string, double>> row = {
            {"update", (double)st.update},
            {"env_steps", (double)st.envSteps},
            {"sps", sps},
            {"elo", elo.current()},
            {"win_vs_bot_train", winVsScripted},
            {"win_vs_pool_train", winVsPool},
            {"ep_len", epCount ? epLenSum / (double)epCount : 0.0},
            {"ep_hits", epCount ? epHitsSum / (double)epCount : 0.0},
            {"ep_crits", epCount ? epCritsSum / (double)epCount : 0.0},
            {"ep_jumps", epCount ? epJumpsSum / (double)epCount : 0.0},
            {"ep_maxcombo", epCount ? epMaxComboSum / (double)epCount : 0.0},
            {"pi_loss", ps.piLoss},
            {"v_loss", ps.vLoss},
            {"entropy", ps.entropy},
            {"kl", ps.kl},
            {"clip_frac", ps.clipFrac},
            {"max_logit", ps.maxLogit},
        };
        for (size_t b = 0; b < ps.entBranch.size(); ++b)
            row.emplace_back("ent_b" + std::to_string(b), ps.entBranch[b]);
        row.emplace_back("lr", p.lr);
        row.emplace_back("ent_coef", p.entCoef);
        row.emplace_back("collect_ms", collectMs);
        row.emplace_back("update_ms", updateMs);
        row.emplace_back("stall_ms", stallMs);
        row.emplace_back("other_ms", otherMs);
        row.emplace_back("pool_size", (double)oppPool.size());
        for (auto& [k, v] : vec.rewardStats()) row.emplace_back(k, v);
        log.log(row);
        if (st.update % logInterval == 0) {
            std::printf("upd %6ld | steps %10ld | sps %8.0f | elo %6.0f | wr(bot) %.2f | "
                        "eplen %5.0f | ent %.3f | kl %.4f | c%4.0f‖u%4.0f +%3.0fms\n",
                        st.update, st.envSteps, sps, elo.current(), winVsScripted,
                        epCount ? epLenSum / (double)epCount : 0.0, ps.entropy, ps.kl, collectMs,
                        updateMs, stallMs);
            epCount = 0;
            epLenSum = epHitsSum = 0;
            epCritsSum = epJumpsSum = epMaxComboSum = 0;
        }
    }

    if (collector.joinable()) collector.join();
    std::printf("stopping: saving checkpoint to %s\n", dir.c_str());
    saveCheckpoint(dir, policy, critic, learner.optimizer(), st);
    oppPool.saveMeta(dir);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
}
