#include "replay.h"

#include <cstdint>
#include <cstring>
#include <fstream>

namespace rl {

namespace {
constexpr char MAGIC[4] = {'M', 'C', 'R', 'P'};
constexpr uint32_t VERSION = 2;
}

bool Replay::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(MAGIC, 4);
    uint32_t v = VERSION, ns = (uint32_t)setup.size(), nt = (uint32_t)ticks.size(),
             np = (uint32_t)pings.size();
    f.write((const char*)&v, 4);
    f.write((const char*)&ns, 4);
    f.write((const char*)&nt, 4);
    f.write((const char*)&np, 4);
    f.write((const char*)setup.data(), (std::streamsize)(ns * sizeof(SetupOp)));
    f.write((const char*)ticks.data(), (std::streamsize)(nt * sizeof(std::array<mc::Input, 2>)));
    f.write((const char*)pings.data(), (std::streamsize)(np * sizeof(std::array<int32_t, 2>)));
    return f.good();
}

std::optional<Replay> Replay::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    char magic[4];
    uint32_t v = 0, ns = 0, nt = 0, np = 0;
    f.read(magic, 4);
    f.read((char*)&v, 4);
    f.read((char*)&ns, 4);
    f.read((char*)&nt, 4);
    if (!f.good() || std::memcmp(magic, MAGIC, 4) != 0 || (v != 1 && v != 2))
        return std::nullopt;
    if (v >= 2) {
        f.read((char*)&np, 4);
        if (!f.good()) return std::nullopt;
    }
    Replay r;
    r.setup.resize(ns);
    r.ticks.resize(nt);
    f.read((char*)r.setup.data(), (std::streamsize)(ns * sizeof(SetupOp)));
    f.read((char*)r.ticks.data(), (std::streamsize)(nt * sizeof(std::array<mc::Input, 2>)));
    if (v >= 2) {
        r.pings.resize(np);
        f.read((char*)r.pings.data(), (std::streamsize)(np * sizeof(std::array<int32_t, 2>)));
    }
    if (!f.good()) return std::nullopt;
    return r;
}

}
