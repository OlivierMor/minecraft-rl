#include "env/setters_builtin.h"

#include <cmath>
#include <stdexcept>

namespace rl {

namespace {
double uni(std::mt19937_64& rng, double lo, double hi) {
    return lo + (hi - lo) * std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

void randomSpot(std::mt19937_64& rng, double margin, double& x, double& z) {
    double r = mc::Arena::RADIUS - margin;
    for (int tries = 0; tries < 128; ++tries) {
        double dx = uni(rng, -r, r), dz = uni(rng, -r, r);
        if (dx * dx + dz * dz <= r * r) {
            x = mc::Arena::CENTER_X + dx;
            z = mc::Arena::CENTER_Z + dz;
            return;
        }
    }
    x = mc::Arena::CENTER_X;
    z = mc::Arena::CENTER_Z;
}
constexpr double CENTER_X = 32.0, CENTER_Z = 32.0;
}

DefaultStateSetter::DefaultStateSetter(const Config& cfg) {
    ping0_ = (int)cfg.num("state_setter.ping0", cfg.num("state_setter.ping", 25));
    ping1_ = (int)cfg.num("state_setter.ping1", cfg.num("state_setter.ping", 25));
}

void DefaultStateSetter::apply(EnvHandle& h, std::mt19937_64& rng) {
    h.reseed(rng());
    h.place(0, CENTER_X - 7.5, 0.0, CENTER_Z, -90.0f, 0.0f);
    h.place(1, CENTER_X + 7.5, 0.0, CENTER_Z, 90.0f, 0.0f);
    h.setPingMs(0, ping0_);
    h.setPingMs(1, ping1_);
}

RandomStateSetter::RandomStateSetter(const Config& cfg) {
    margin_ = cfg.num("state_setter.margin", 1.5);
    minDist_ = cfg.num("state_setter.min_dist", 3.0);
    maxDist_ = cfg.num("state_setter.max_dist", 22.0);
    pingMin_ = (int)cfg.num("state_setter.ping_min", 0);
    pingMax_ = (int)cfg.num("state_setter.ping_max", 100);
    pingPow_ = cfg.num("state_setter.ping_pow", 1.0);
    if (pingPow_ < 1.0) pingPow_ = 1.0;
    for (const Value& p : cfg.arr("state_setter.ping_quantiles")) {
        if (p.kind != Value::Arr || p.a.size() != 2 || p.a[0].kind != Value::Num ||
            p.a[1].kind != Value::Num)
            throw std::runtime_error(
                "state_setter.ping_quantiles: expected [[quantile, ping_ms], ...]");
        if (!pingQ_.empty() && p.a[0].n < pingQ_.back().first)
            throw std::runtime_error("state_setter.ping_quantiles: quantiles must be ascending");
        pingQ_.emplace_back(p.a[0].n, p.a[1].n);
    }
    faceEachOther_ = cfg.boolean("state_setter.face_each_other", true);
    yawJitter_ = cfg.num("state_setter.yaw_jitter", 25.0);
}

void RandomStateSetter::apply(EnvHandle& h, std::mt19937_64& rng) {
    h.reseed(rng());
    double x0, z0, x1, z1;
    randomSpot(rng, margin_, x0, z0);
    x1 = x0;
    z1 = z0;
    for (int tries = 0; tries < 64; ++tries) {
        randomSpot(rng, margin_, x1, z1);
        double d = std::hypot(x1 - x0, z1 - z0);
        if (d >= minDist_ && d <= maxDist_) break;
    }
    auto yawToward = [](double fx, double fz, double tx, double tz) {
        return (float)(std::atan2(tz - fz, tx - fx) * 180.0 / M_PI) - 90.0f;
    };
    float y0, y1;
    if (faceEachOther_) {
        y0 = yawToward(x0, z0, x1, z1) + (float)uni(rng, -yawJitter_, yawJitter_);
        y1 = yawToward(x1, z1, x0, z0) + (float)uni(rng, -yawJitter_, yawJitter_);
    } else {
        y0 = (float)uni(rng, -180.0, 180.0);
        y1 = (float)uni(rng, -180.0, 180.0);
    }
    h.place(0, x0, 0.0, z0, y0, 0.0f);
    h.place(1, x1, 0.0, z1, y1, 0.0f);
    auto drawPing = [&]() {
        double u = uni(rng, 0.0, 1.0);
        if (!pingQ_.empty()) {
            double px = pingQ_.front().first, py = pingQ_.front().second;
            double v = pingQ_.back().second;
            if (u <= px) v = py;
            else
                for (auto& [cx, cy] : pingQ_) {
                    if (u <= cx) {
                        v = (cx - px < 1e-12) ? cy : py + (cy - py) * (u - px) / (cx - px);
                        break;
                    }
                    px = cx;
                    py = cy;
                }
            return (int)std::lround(v);
        }
        int p = pingMin_ + (int)((double)(pingMax_ + 1 - pingMin_) * std::pow(u, pingPow_));
        return p > pingMax_ ? pingMax_ : p;
    };
    h.setPingMs(0, drawPing());
    h.setPingMs(1, drawPing());
}

}
