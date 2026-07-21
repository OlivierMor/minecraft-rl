#pragma once
#include <cassert>
#include "geom.h"

namespace mc1218 {

struct BlockRayHit {
    Vec3 hitVec;
};

struct BoxList {
    static constexpr int CAP = 160;
    AABB box[CAP];
    int n = 0;

    void clear() { n = 0; }
    void push_back(const AABB& b) {
        assert(n < CAP);
        if (n < CAP) box[n++] = b;
    }
    const AABB* begin() const { return box; }
    const AABB* end() const { return box + n; }
    bool empty() const { return n == 0; }
};

class Arena {
public:
    static constexpr int DIAMETER = 64;
    static constexpr double RADIUS = DIAMETER / 2.0;
    static constexpr double CENTER_X = DIAMETER / 2.0;
    static constexpr double CENTER_Z = DIAMETER / 2.0;
    static constexpr int WALL_HEIGHT = 4;
    static constexpr int RING = 2;

    static bool insideDisc(int bx, int bz) {
        double dx = (double)bx + 0.5 - CENTER_X;
        double dz = (double)bz + 0.5 - CENTER_Z;
        return dx * dx + dz * dz < RADIUS * RADIUS;
    }
    static bool wallRing(int bx, int bz) {
        if (insideDisc(bx, bz)) return false;
        double dx = (double)bx + 0.5 - CENTER_X;
        double dz = (double)bz + 0.5 - CENTER_Z;
        double r = RADIUS + (double)RING;
        return dx * dx + dz * dz < r * r;
    }

    bool solid(int bx, int by, int bz) const {
        if (by == -1) return insideDisc(bx, bz) || wallRing(bx, bz);
        if (by >= 0 && by < WALL_HEIGHT) return wallRing(bx, bz);
        return false;
    }

    static bool deepInsideDisc(const AABB& q) {
        double dx = std::max(std::abs(q.minX - CENTER_X), std::abs(q.maxX - CENTER_X));
        double dz = std::max(std::abs(q.minZ - CENTER_Z), std::abs(q.maxZ - CENTER_Z));
        double lim = RADIUS - 0.75;
        return dx * dx + dz * dz < lim * lim;
    }

    void getCollidingBoundingBoxes(const AABB& query, BoxList& out) const {
        out.clear();
        int x0 = Mth::floor(query.minX) - 1, x1 = Mth::floor(query.maxX) + 1;
        int y0 = Mth::floor(query.minY) - 1, y1 = Mth::floor(query.maxY) + 1;
        int z0 = Mth::floor(query.minZ) - 1, z1 = Mth::floor(query.maxZ) + 1;
        if (y0 > WALL_HEIGHT || y1 < -1) return;
        if (y0 < -1) y0 = -1;
        if (y1 > WALL_HEIGHT - 1) y1 = WALL_HEIGHT - 1;
        if (deepInsideDisc(query)) {
            if (!(query.minY < 0.0 && query.maxY > -1.0)) return;
            for (int x = x0; x <= x1; ++x)
                for (int z = z0; z <= z1; ++z) {
                    AABB b((double)x, -1.0, (double)z, (double)x + 1.0, 0.0, (double)z + 1.0);
                    if (query.intersects(b)) out.push_back(b);
                }
            return;
        }
        for (int x = x0; x <= x1; ++x)
            for (int z = z0; z <= z1; ++z)
                for (int y = y0; y <= y1; ++y) {
                    if (!solid(x, y, z)) continue;
                    AABB b((double)x, (double)y, (double)z,
                           (double)x + 1.0, (double)y + 1.0, (double)z + 1.0);
                    if (query.intersects(b)) out.push_back(b);
                }
    }

    bool noCollision(const AABB& query) const {
        int x0 = Mth::floor(query.minX) - 1, x1 = Mth::floor(query.maxX) + 1;
        int y0 = Mth::floor(query.minY) - 1, y1 = Mth::floor(query.maxY) + 1;
        int z0 = Mth::floor(query.minZ) - 1, z1 = Mth::floor(query.maxZ) + 1;
        if (y0 > WALL_HEIGHT || y1 < -1) return true;
        if (y0 < -1) y0 = -1;
        if (y1 > WALL_HEIGHT - 1) y1 = WALL_HEIGHT - 1;
        if (deepInsideDisc(query))
            return !(query.minY < 0.0 && query.maxY > -1.0);
        for (int x = x0; x <= x1; ++x)
            for (int z = z0; z <= z1; ++z)
                for (int y = y0; y <= y1; ++y) {
                    if (!solid(x, y, z)) continue;
                    AABB b((double)x, (double)y, (double)z,
                           (double)x + 1.0, (double)y + 1.0, (double)z + 1.0);
                    if (query.intersects(b)) return false;
                }
        return true;
    }

    std::optional<BlockRayHit> rayTraceBlocks(const Vec3& from, const Vec3& to) const {
        int bx = Mth::floor(from.x), by = Mth::floor(from.y), bz = Mth::floor(from.z);
        if (solid(bx, by, bz)) return BlockRayHit{from};

        double dx = to.x - from.x, dy = to.y - from.y, dz = to.z - from.z;
        int sx = dx > 0 ? 1 : (dx < 0 ? -1 : 0);
        int sy = dy > 0 ? 1 : (dy < 0 ? -1 : 0);
        int sz = dz > 0 ? 1 : (dz < 0 ? -1 : 0);
        auto boundary = [](double p, int b, int s) {
            return s > 0 ? (double)(b + 1) - p : p - (double)b;
        };
        double tMaxX = sx != 0 ? boundary(from.x, bx, sx) / std::abs(dx) : 1e300;
        double tMaxY = sy != 0 ? boundary(from.y, by, sy) / std::abs(dy) : 1e300;
        double tMaxZ = sz != 0 ? boundary(from.z, bz, sz) / std::abs(dz) : 1e300;
        double tDeltaX = sx != 0 ? 1.0 / std::abs(dx) : 1e300;
        double tDeltaY = sy != 0 ? 1.0 / std::abs(dy) : 1e300;
        double tDeltaZ = sz != 0 ? 1.0 / std::abs(dz) : 1e300;

        for (int it = 0; it < 400; ++it) {
            double t;
            if (tMaxX <= tMaxY && tMaxX <= tMaxZ) { t = tMaxX; tMaxX += tDeltaX; bx += sx; }
            else if (tMaxY <= tMaxZ)              { t = tMaxY; tMaxY += tDeltaY; by += sy; }
            else                                  { t = tMaxZ; tMaxZ += tDeltaZ; bz += sz; }
            if (t > 1.0) return std::nullopt;
            if (solid(bx, by, bz))
                return BlockRayHit{Vec3{from.x + dx * t, from.y + dy * t, from.z + dz * t}};
        }
        return std::nullopt;
    }
};

}
