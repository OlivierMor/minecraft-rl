#include "ppo/learner.h"

#include <cstdio>

#if __has_include(<ATen/autocast_mode.h>)
#include <ATen/autocast_mode.h>
#define MCRL_HAS_AUTOCAST 1
#else
#define MCRL_HAS_AUTOCAST 0
#endif

#include "model/distributions.h"

namespace rl {

PPOParams PPOParams::fromConfig(const Config& cfg) {
    PPOParams p;
    p.gamma = (float)cfg.num("ppo.gamma", 0.99);
    p.lam = (float)cfg.num("ppo.lam", 0.95);
    p.clip = (float)cfg.num("ppo.clip", 0.2);
    p.vfCoef = (float)cfg.num("ppo.vf_coef", 0.5);
    p.gradClip = (float)cfg.num("ppo.grad_clip", 0.5);
    p.klStop = (float)cfg.num("ppo.kl_stop", 0.0);
    p.vClip = (float)cfg.num("ppo.v_clip", 0.0);
    p.dualClip = (float)cfg.num("ppo.dual_clip", 3.0);
    p.logitL2 = (float)cfg.num("ppo.logit_l2", 0.001);
    p.epochs = (int)cfg.num("ppo.epochs", 3);
    p.minibatch = (int)cfg.num("ppo.minibatch", 65536);
    p.normAdv = cfg.boolean("ppo.norm_adv", true);
    p.lr = (float)cfg.num("ppo.lr", 3e-4);
    p.entCoef = (float)cfg.num("ppo.entropy", 0.01);
    return p;
}

Learner::Learner(PolicyNet policy, CriticNet critic, torch::Device device, const Config& cfg)
    : policy_(std::move(policy)), critic_(std::move(critic)),
      hasCritic_(!critic_.is_empty()), device_(device),
      gpuBatch_(cfg.boolean("ppo.gpu_batch", true)),
      bf16_(cfg.boolean("ppo.bf16", false) && device.is_cuda()) {
    policy_->to(device_);
    allParams_ = policy_->parameters();
    if (hasCritic_) {
        critic_->to(device_);
        for (auto& p : critic_->parameters()) allParams_.push_back(p);
    }
    opt_ = std::make_unique<torch::optim::Adam>(
        allParams_, torch::optim::AdamOptions((double)cfg.num("ppo.lr", 3e-4)).eps(1e-5));
}

void Learner::setLr(double lr) {
    for (auto& g : opt_->param_groups())
        static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr);
}

PPOStats Learner::update(Rollout& roll, const torch::Tensor& advIn, const torch::Tensor& retIn,
                         const PPOParams& p) {
    const int64_t B = (int64_t)roll.T * roll.A;
    torch::Tensor obs = roll.obs.view({B, roll.obsDim});
    torch::Tensor cobs = hasCritic_ ? roll.cobs.view({B, roll.criticObsDim}) : torch::Tensor();
    torch::Tensor act = roll.actions.view({B, roll.numBranches});
    torch::Tensor oldLogp = roll.logp.view({B});
    torch::Tensor oldVal = roll.value.view({B});
    torch::Tensor adv = advIn.view({B});
    torch::Tensor ret = retIn.view({B});

    if (p.normAdv) adv = (adv - adv.mean()) / (adv.std() + 1e-8);

    bool onDevice = false;
    if (gpuBatch_ && device_.is_cuda()) {
        try {
            auto batchDt = bf16_ ? torch::kBFloat16 : torch::kFloat32;
            torch::Tensor obsD = obs.to(device_, batchDt, true);
            torch::Tensor cobsD = hasCritic_ ? cobs.to(device_, batchDt, true) : torch::Tensor();
            torch::Tensor actD = act.to(device_, true);
            torch::Tensor lpD = oldLogp.to(device_, true);
            torch::Tensor ovD = oldVal.to(device_, true);
            torch::Tensor advD = adv.to(device_, true);
            torch::Tensor retD = ret.to(device_, true);
            obs = obsD; cobs = cobsD; act = actD;
            oldLogp = lpD; oldVal = ovD; adv = advD; ret = retD;
            onDevice = true;
        } catch (const c10::Error&) {
            std::fprintf(stderr,
                         "learner: batch does not fit on the GPU - using CPU gathers\n");
            gpuBatch_ = false;
        }
    }

    setLr(p.lr);
    PPOStats st;
    int64_t mb = std::min<int64_t>(p.minibatch, B);
    int64_t nMb = (B + mb - 1) / mb;
    long samples = 0;
    bool bail = false;

    for (int epoch = 0; epoch < p.epochs && !bail; ++epoch) {
        torch::Tensor perm = onDevice
            ? torch::randperm(B, torch::TensorOptions().dtype(torch::kInt64).device(device_))
            : torch::randperm(B, torch::kInt64);
        double epochKl = 0;
        int64_t klCount = 0;
        for (int64_t m = 0; m < nMb; ++m) {
            torch::Tensor idx = perm.narrow(0, m * mb, std::min(mb, B - m * mb));
            if (onDevice) idx = idx.to(device_, true);
            auto take = [&](const torch::Tensor& t) {
                torch::Tensor sel = t.index_select(0, idx);
                return onDevice ? sel : sel.to(device_, true);
            };
            torch::Tensor o = take(obs);
            torch::Tensor a = take(act);
            torch::Tensor lp0 = take(oldLogp);
            torch::Tensor v0 = take(oldVal);
            torch::Tensor ad = take(adv);
            torch::Tensor rt = take(ret);

#if MCRL_HAS_AUTOCAST
            if (bf16_) {
                at::autocast::set_autocast_enabled(at::kCUDA, true);
                at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
            }
#endif
            torch::Tensor logits, val;
            if (hasCritic_) {
                torch::Tensor co = take(cobs);
                logits = policy_->logitsOnly(o);
                val = critic_->forward(co);
            } else {
                std::tie(logits, val) = policy_->forward(o);
            }
            logits = logits.to(torch::kFloat32);
            val = val.to(torch::kFloat32);
#if MCRL_HAS_AUTOCAST
            if (bf16_) {
                at::autocast::clear_cache();
                at::autocast::set_autocast_enabled(at::kCUDA, false);
            }
#endif
            auto [newLogp, entropy] = logProbEntropy(logits, policy_->spec, a.to(torch::kInt64));
            if (st.entBranch.empty()) st.entBranch = branchEntropies(logits, policy_->spec);

            torch::Tensor logRatio = newLogp - lp0;
            torch::Tensor ratio = logRatio.exp();
            torch::Tensor s1 = -ad * ratio;
            torch::Tensor s2 = -ad * ratio.clamp(1.0f - p.clip, 1.0f + p.clip);
            torch::Tensor perSample = torch::max(s1, s2);
            if (p.dualClip > 0) {
                torch::Tensor cap = -p.dualClip * ad;
                perSample = torch::where(ad < 0, torch::min(perSample, cap), perSample);
            }
            torch::Tensor piLoss = perSample.mean();

            torch::Tensor vLoss;
            if (p.vClip > 0) {
                torch::Tensor vClipped = v0 + (val - v0).clamp(-p.vClip, p.vClip);
                vLoss = 0.5f * torch::max((val - rt).pow(2), (vClipped - rt).pow(2)).mean();
            } else {
                vLoss = 0.5f * (val - rt).pow(2).mean();
            }
            torch::Tensor entMean = entropy.mean();
            torch::Tensor loss = piLoss + p.vfCoef * vLoss - p.entCoef * entMean;
            if (p.logitL2 > 0) loss = loss + p.logitL2 * logits.pow(2).sum(-1).mean();

            opt_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(allParams_, p.gradClip);
            opt_->step();

            {
                torch::NoGradGuard ng;
                torch::Tensor lrC = logRatio.clamp(-10.0f, 10.0f);
                torch::Tensor packed = torch::stack({((lrC.exp() - 1.0f) - lrC).mean(),
                                                     (ratio - 1.0f).abs().gt(p.clip)
                                                         .to(torch::kFloat32).mean(),
                                                     piLoss, vLoss, entMean,
                                                     logits.abs().max()})
                                           .to(torch::kCPU);
                auto pk = packed.accessor<float, 1>();
                double kl = pk[0];
                st.kl += kl;
                st.clipFrac += pk[1];
                st.piLoss += pk[2];
                st.vLoss += pk[3];
                st.entropy += pk[4];
                st.maxLogit = std::max(st.maxLogit, (double)pk[5]);
                epochKl += kl;
                ++klCount;
                ++samples;
                if (p.klStop > 0 && kl > 2.0 * p.klStop) bail = true;
            }
            if (bail) break;
        }
        ++st.epochsRan;
        if (p.klStop > 0 && klCount > 0 && epochKl / (double)klCount > p.klStop) break;
    }
    if (samples > 0) {
        st.piLoss /= (double)samples;
        st.vLoss /= (double)samples;
        st.entropy /= (double)samples;
        st.kl /= (double)samples;
        st.clipFrac /= (double)samples;
    }
    return st;
}

}
