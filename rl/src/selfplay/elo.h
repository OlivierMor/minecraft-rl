#pragma once
#include <string>

namespace rl {

class EloTracker {
public:
    EloTracker(double anchorElo, double kCurrent, double kPool)
        : anchor_(anchorElo), current_(anchorElo), kCur_(kCurrent), kPool_(kPool) {}

    static double expected(double ra, double rb);

    double current() const { return current_; }
    void setCurrent(double e) { current_ = e; }
    double anchor() const { return anchor_; }

    double applyResult(double oppElo, bool oppIsAnchor, double score);

    void openMatchLog(const std::string& csvPath);
    void logMatch(long update, const std::string& opp, double oppElo, double score);

private:
    double anchor_, current_, kCur_, kPool_;
    std::string csvPath_;
};

}
