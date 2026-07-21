#include "env/registry.h"

#include <algorithm>
#include <stdexcept>

#include "env/rewards_builtin.h"
#include "env/setters_builtin.h"
#include "env/standard_actions.h"
#include "env/standard_obs.h"
#include "env/terminals_builtin.h"

namespace rl {
namespace registry {

std::unique_ptr<ObsBuilder> makeObs(const std::string& name, const Config& cfg, int side) {
    if (name == "standard") return std::make_unique<StandardObs>(cfg, side);
    throw std::runtime_error("unknown obs builder: '" + name + "'");
}

std::unique_ptr<CriticObsBuilder> makeCriticObs(const std::string& name, const Config& cfg) {
    if (name == "standard_privileged") return std::make_unique<StandardCriticObs>(cfg);
    throw std::runtime_error("unknown critic obs builder: '" + name + "'");
}

std::unique_ptr<ActionParser> makeParser(const std::string& name, const Config& cfg) {
    if (name == "standard") return std::make_unique<StandardActions>(cfg);
    throw std::runtime_error("unknown action parser: '" + name + "'");
}

std::unique_ptr<RewardFunction> makeReward(const std::string& name, const Config& cfg) {
    if (name == "hit") return std::make_unique<HitReward>();
    if (name == "crit") return std::make_unique<CritReward>();
    if (name == "hit_taken") return std::make_unique<HitTakenReward>();
    if (name == "win_loss") return std::make_unique<WinLossReward>();
    if (name == "approach") return std::make_unique<ApproachReward>(cfg);
    if (name == "look_far") return std::make_unique<LookFarReward>(cfg);
    if (name == "aim") return std::make_unique<AimReward>();
    if (name == "combo_exp") return std::make_unique<ComboExpReward>(cfg);
    if (name == "crit_combo") return std::make_unique<CritComboReward>(cfg);
    if (name == "time") return std::make_unique<TimeReward>();
    if (name == "draw") return std::make_unique<DrawReward>();
    if (name == "jump") return std::make_unique<JumpReward>();
    if (name == "sneak") return std::make_unique<SneakReward>();
    if (name == "strafe") return std::make_unique<StrafeReward>(cfg);
    if (name == "aim_pressure") return std::make_unique<AimPressureReward>();
    if (name == "cursor_pressure") return std::make_unique<CursorPressureReward>(cfg);
    if (name == "aim_jerk") return std::make_unique<AimJerkReward>();
    if (name == "fast_win") return std::make_unique<FastWinReward>(cfg);
    if (name == "fight_clock") return std::make_unique<FightClockReward>(cfg);
    if (name == "missed_swing") return std::make_unique<MissedSwingReward>();
    if (name == "early_swing") return std::make_unique<EarlySwingReward>(cfg);
    if (name == "damage_dealt") return std::make_unique<DamageDealtReward>();
    if (name == "damage_taken") return std::make_unique<DamageTakenReward>();
    if (name == "combined") {
        auto combined = std::make_unique<CombinedReward>();
        auto comps = cfg.arr("reward.components");
        if (comps.empty()) {
            combined->add("damage_dealt", 1.0f, makeReward("damage_dealt", cfg));
            combined->add("damage_taken", -1.0f, makeReward("damage_taken", cfg));
            combined->add("win_loss", 10.0f, makeReward("win_loss", cfg));
        } else {
            std::vector<std::string> labels;
            auto uniqueLabel = [&](std::string l, bool shaped) {
                if (std::count(labels.begin(), labels.end(), l)) {
                    if (shaped && !std::count(labels.begin(), labels.end(), l + "_anneal")) {
                        l += "_anneal";
                    } else {
                        int k = 2;
                        while (std::count(labels.begin(), labels.end(),
                                          l + "_" + std::to_string(k)))
                            ++k;
                        l += "_" + std::to_string(k);
                    }
                }
                labels.push_back(l);
                return l;
            };
            for (const Value& c : comps) {
                bool ok = c.kind == Value::Arr && c.a.size() >= 2 && c.a.size() <= 3 &&
                          c.a[0].kind == Value::Str && c.a[1].kind == Value::Num &&
                          (c.a.size() == 2 || c.a[2].kind == Value::Str);
                if (!ok)
                    throw std::runtime_error(
                        "reward.components: expected [[\"name\", weight] or "
                        "[\"name\", weight, \"anneal\"], ...]");
                bool shaped = c.a.size() == 3 && (c.a[2].s == "anneal" || c.a[2].s == "shaped");
                combined->add(uniqueLabel(c.a[0].s, shaped), (float)c.a[1].n,
                              makeReward(c.a[0].s, cfg), shaped);
            }
        }
        auto sched = cfg.arr("reward.shaping_schedule");
        if (!sched.empty()) {
            std::vector<std::pair<double, double>> pts;
            for (const Value& p : sched) {
                if (p.kind != Value::Arr || p.a.size() != 2)
                    throw std::runtime_error("reward.shaping_schedule: expected [[x,y],...]");
                pts.emplace_back(p.a[0].n, p.a[1].n);
            }
            combined->setShapingSchedule(std::move(pts));
        }
        if (cfg.boolean("reward.zero_sum", false))
            return std::make_unique<ZeroSumReward>(std::move(combined));
        return combined;
    }
    throw std::runtime_error("unknown reward function: '" + name + "'");
}

std::unique_ptr<TerminalCondition> makeTerminal(const std::string& name, const Config& cfg) {
    if (name == "first_to_hits") return std::make_unique<FirstToHits>(cfg);
    if (name == "tick_cap") return std::make_unique<TickCap>(cfg);
    if (name == "no_hit_timeout") return std::make_unique<NoHitTimeout>(cfg);
    if (name == "death") return std::make_unique<DeathMatch>(cfg);
    throw std::runtime_error("unknown terminal condition: '" + name + "'");
}

std::unique_ptr<StateSetter> makeSetter(const std::string& name, const Config& cfg) {
    if (name == "default") return std::make_unique<DefaultStateSetter>(cfg);
    if (name == "random") return std::make_unique<RandomStateSetter>(cfg);
    throw std::runtime_error("unknown state setter: '" + name + "'");
}

ComponentSet makeComponents(const Config& cfg) {
    std::string env = cfg.str("env", MC_ENV_NAME);
    if (env != MC_ENV_NAME)
        throw std::runtime_error("config says env = '" + env + "' but this binary is built for '" +
                                 MC_ENV_NAME "' - use the matching tool (train189/train1218, ...)");

    ComponentSet c;
    std::string obs = cfg.str("components.obs", "standard");
    std::string act = cfg.str("components.actions", "standard");
    std::string rew = cfg.str("components.reward", "combined");
    std::string set = cfg.str("components.state_setter", "random");
    std::string critic = cfg.str("components.critic_obs", "");
    auto terms = cfg.strArr("components.terminal");
    if (terms.empty()) terms = {"death", "tick_cap"};

    for (int s = 0; s < 2; ++s) {
        c.obs[s] = makeObs(obs, cfg, s);
        c.parser[s] = makeParser(act, cfg);
    }
    if (!critic.empty()) c.criticObs = makeCriticObs(critic, cfg);
    c.reward = makeReward(rew, cfg);
    for (const std::string& t : terms) c.terminals.push_back(makeTerminal(t, cfg));
    c.setter = makeSetter(set, cfg);
    return c;
}

}
}
