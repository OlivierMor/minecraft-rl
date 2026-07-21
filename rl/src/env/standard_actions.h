#pragma once
#include "config.h"
#include "env/interfaces.h"

namespace rl {

class StandardActions : public ActionParser {
public:
    explicit StandardActions(const Config& cfg);
    ActionSpec spec() const override;
    void reset() override;
    mc::Input parse(const int32_t* b) override;

private:
    std::vector<float> yaw_, pitch_;
    bool allowSneak_;
    float yawAccel_ = 0, pitchAccel_ = 0;
    float prevYaw_ = 0, prevPitch_ = 0;
};

}
