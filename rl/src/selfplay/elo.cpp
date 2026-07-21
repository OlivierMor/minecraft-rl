#include "selfplay/elo.h"

#include <cmath>
#include <filesystem>
#include <fstream>

namespace rl {

double EloTracker::expected(double ra, double rb) {
    return 1.0 / (1.0 + std::pow(10.0, (rb - ra) / 400.0));
}

double EloTracker::applyResult(double oppElo, bool oppIsAnchor, double score) {
    double e = expected(current_, oppElo);
    current_ += kCur_ * (score - e);
    if (oppIsAnchor) return oppElo;
    return oppElo + kPool_ * ((1.0 - score) - expected(oppElo, current_));
}

void EloTracker::openMatchLog(const std::string& csvPath) {
    csvPath_ = csvPath;
    if (!std::filesystem::exists(csvPath_)) {
        std::ofstream f(csvPath_);
        f << "update,opponent,opp_elo,score,current_elo\n";
    }
}

void EloTracker::logMatch(long update, const std::string& opp, double oppElo, double score) {
    if (csvPath_.empty()) return;
    std::ofstream f(csvPath_, std::ios::app);
    f << update << "," << opp << "," << oppElo << "," << score << "," << current_ << "\n";
}

}
