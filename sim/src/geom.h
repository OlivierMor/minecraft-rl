#pragma once
#include <optional>
#include "mc_math.h"

namespace mc1218 {

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 add(double dx, double dy, double dz) const { return {x + dx, y + dy, z + dz}; }
    Vec3 add(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 subtract(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 scale(double d) const { return {x * d, y * d, z * d}; }
    Vec3 multiply(double dx, double dy, double dz) const { return {x * dx, y * dy, z * dz}; }

    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    double lengthSqr() const { return x * x + y * y + z * z; }
    double length() const { return std::sqrt(lengthSqr()); }
    double horizontalDistance() const { return std::sqrt(x * x + z * z); }
    double horizontalDistanceSqr() const { return x * x + z * z; }
    double distanceToSqr(const Vec3& v) const {
        double d = v.x - x, d2 = v.y - y, d3 = v.z - z;
        return d * d + d2 * d2 + d3 * d3;
    }
    double distanceTo(const Vec3& v) const { return std::sqrt(distanceToSqr(v)); }

    Vec3 normalize() const {
        double d = std::sqrt(x * x + y * y + z * z);
        if (d < (double)1.0E-4f) return {0.0, 0.0, 0.0};
        return {x / d, y / d, z / d};
    }

    bool closerThan(const Vec3& v, double d) const { return distanceToSqr(v) < d * d; }
};

struct Vec2 {
    float x = 0.0f, y = 0.0f;
    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}
    Vec2 scale(float f) const { return {x * f, y * f}; }
    float lengthSquared() const { return x * x + y * y; }
    float length() const { return Mth::sqrt(x * x + y * y); }
    Vec2 normalized() const {
        float f = Mth::sqrt(x * x + y * y);
        return f < 1.0E-4f ? Vec2{} : Vec2{x / f, y / f};
    }
};

struct AABB {
    double minX = 0, minY = 0, minZ = 0, maxX = 0, maxY = 0, maxZ = 0;

    AABB() = default;
    AABB(double x1, double y1, double z1, double x2, double y2, double z2)
        : minX(std::min(x1, x2)), minY(std::min(y1, y2)), minZ(std::min(z1, z2)),
          maxX(std::max(x1, x2)), maxY(std::max(y1, y2)), maxZ(std::max(z1, z2)) {}

    static AABB raw(double x1, double y1, double z1, double x2, double y2, double z2) {
        AABB r;
        r.minX = x1; r.minY = y1; r.minZ = z1;
        r.maxX = x2; r.maxY = y2; r.maxZ = z2;
        return r;
    }

    AABB expandTowards(double dx, double dy, double dz) const {
        double x1 = minX, y1 = minY, z1 = minZ, x2 = maxX, y2 = maxY, z2 = maxZ;
        if (dx < 0.0) x1 += dx; else if (dx > 0.0) x2 += dx;
        if (dy < 0.0) y1 += dy; else if (dy > 0.0) y2 += dy;
        if (dz < 0.0) z1 += dz; else if (dz > 0.0) z2 += dz;
        return raw(x1, y1, z1, x2, y2, z2);
    }
    AABB inflate(double dx, double dy, double dz) const {
        return raw(minX - dx, minY - dy, minZ - dz, maxX + dx, maxY + dy, maxZ + dz);
    }
    AABB inflate(double d) const { return inflate(d, d, d); }
    AABB deflate(double d) const { return inflate(-d); }
    AABB expand(double dx, double dy, double dz) const { return inflate(dx, dy, dz); }
    AABB move(double dx, double dy, double dz) const {
        return raw(minX + dx, minY + dy, minZ + dz, maxX + dx, maxY + dy, maxZ + dz);
    }

    bool intersects(const AABB& o) const {
        return minX < o.maxX && maxX > o.minX && minY < o.maxY && maxY > o.minY &&
               minZ < o.maxZ && maxZ > o.minZ;
    }
    bool contains(const Vec3& v) const {
        return v.x >= minX && v.x < maxX && v.y >= minY && v.y < maxY && v.z >= minZ && v.z < maxZ;
    }

    Vec3 getCenter() const { return {(minX + maxX) / 2.0, (minY + maxY) / 2.0, (minZ + maxZ) / 2.0}; }
    Vec3 getBottomCenter() const { return {(minX + maxX) / 2.0, minY, (minZ + maxZ) / 2.0}; }

    double distanceToSqr(const Vec3& v) const {
        double d = std::max(std::max(minX - v.x, v.x - maxX), 0.0);
        double d2 = std::max(std::max(minY - v.y, v.y - maxY), 0.0);
        double d3 = std::max(std::max(minZ - v.z, v.z - maxZ), 0.0);
        return Mth::lengthSquared(d, d2, d3);
    }

    std::optional<Vec3> clip(const Vec3& from, const Vec3& to) const {
        double dx = to.x - from.x, dy = to.y - from.y, dz = to.z - from.z;
        double tBest = 1.0;
        bool found = false;
        auto axisCheck = [&](double dA, double minA, double maxA, double fromA,
                             double dB, double fromB, double minB, double maxB,
                             double dC, double fromC, double minC, double maxC) {
            if (dA > 1.0E-7) {
                double t = (minA - fromA) / dA;
                if (t >= 0.0 && t < tBest) {
                    double b = fromB + dB * t, c = fromC + dC * t;
                    if (b >= minB && b <= maxB && c >= minC && c <= maxC) { tBest = t; found = true; }
                }
            } else if (dA < -1.0E-7) {
                double t = (maxA - fromA) / dA;
                if (t >= 0.0 && t < tBest) {
                    double b = fromB + dB * t, c = fromC + dC * t;
                    if (b >= minB && b <= maxB && c >= minC && c <= maxC) { tBest = t; found = true; }
                }
            }
        };
        axisCheck(dx, minX, maxX, from.x, dy, from.y, minY, maxY, dz, from.z, minZ, maxZ);
        axisCheck(dy, minY, maxY, from.y, dz, from.z, minZ, maxZ, dx, from.x, minX, maxX);
        axisCheck(dz, minZ, maxZ, from.z, dx, from.x, minX, maxX, dy, from.y, minY, maxY);
        if (!found) return std::nullopt;
        return Vec3{from.x + tBest * dx, from.y + tBest * dy, from.z + tBest * dz};
    }
};


inline bool cubeOverlapPerp(double boxMin, double boxMax, double cubeMin, double cubeMax) {
    return boxMin + 1.0E-7 < cubeMax && boxMax - 1.0E-7 >= cubeMin;
}

inline double cubeCollideX(const AABB& cube, const AABB& box, double d) {
    if (!cubeOverlapPerp(box.minY, box.maxY, cube.minY, cube.maxY)) return d;
    if (!cubeOverlapPerp(box.minZ, box.maxZ, cube.minZ, cube.maxZ)) return d;
    if (d > 0.0) {
        if (box.maxX - 1.0E-7 < cube.minX) d = std::min(d, cube.minX - box.maxX);
    } else if (d < 0.0) {
        if (box.minX + 1.0E-7 > cube.maxX) d = std::max(d, cube.maxX - box.minX);
    }
    return d;
}
inline double cubeCollideY(const AABB& cube, const AABB& box, double d) {
    if (!cubeOverlapPerp(box.minZ, box.maxZ, cube.minZ, cube.maxZ)) return d;
    if (!cubeOverlapPerp(box.minX, box.maxX, cube.minX, cube.maxX)) return d;
    if (d > 0.0) {
        if (box.maxY - 1.0E-7 < cube.minY) d = std::min(d, cube.minY - box.maxY);
    } else if (d < 0.0) {
        if (box.minY + 1.0E-7 > cube.maxY) d = std::max(d, cube.maxY - box.minY);
    }
    return d;
}
inline double cubeCollideZ(const AABB& cube, const AABB& box, double d) {
    if (!cubeOverlapPerp(box.minX, box.maxX, cube.minX, cube.maxX)) return d;
    if (!cubeOverlapPerp(box.minY, box.maxY, cube.minY, cube.maxY)) return d;
    if (d > 0.0) {
        if (box.maxZ - 1.0E-7 < cube.minZ) d = std::min(d, cube.minZ - box.maxZ);
    } else if (d < 0.0) {
        if (box.minZ + 1.0E-7 > cube.maxZ) d = std::max(d, cube.maxZ - box.minZ);
    }
    return d;
}

}
