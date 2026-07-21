#pragma once
#include <memory>
#include <string>

#include "config.h"
#include "env/env.h"
#include "env/interfaces.h"

namespace rl {

namespace registry {

std::unique_ptr<ObsBuilder> makeObs(const std::string& name, const Config& cfg, int side);
std::unique_ptr<CriticObsBuilder> makeCriticObs(const std::string& name, const Config& cfg);
std::unique_ptr<ActionParser> makeParser(const std::string& name, const Config& cfg);
std::unique_ptr<RewardFunction> makeReward(const std::string& name, const Config& cfg);
std::unique_ptr<TerminalCondition> makeTerminal(const std::string& name, const Config& cfg);
std::unique_ptr<StateSetter> makeSetter(const std::string& name, const Config& cfg);

ComponentSet makeComponents(const Config& cfg);

}

}
