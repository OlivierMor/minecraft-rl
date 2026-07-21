#pragma once
#include <array>
#include <cmath>
#include <cstdint>

namespace mc1218 {

inline constexpr float PI_F = 3.14159265358979323846f;
inline constexpr float DEG_TO_RAD_F = PI_F / 180.0f;

inline int javaIntCast(float f) {
    if (std::isnan(f)) return 0;
    if (f >= 2147483647.0f) return 2147483647;
    if (f <= -2147483648.0f) return -2147483647 - 1;
    return static_cast<int>(f);
}
inline int javaIntCast(double d) {
    if (std::isnan(d)) return 0;
    if (d >= 2147483647.0) return 2147483647;
    if (d <= -2147483648.0) return -2147483647 - 1;
    return static_cast<int>(d);
}
inline int64_t javaLongCast(double d) {
    if (std::isnan(d)) return 0;
    if (d >= 9223372036854775807.0) return INT64_MAX;
    if (d <= -9223372036854775808.0) return INT64_MIN;
    return static_cast<int64_t>(d);
}
inline int64_t javaRound(double d) { return javaLongCast(std::floor(d + 0.5)); }

namespace detail {
inline std::array<float, 65536> makeSinTable() {
    std::array<float, 65536> t{};
    for (int i = 0; i < 65536; ++i)
        t[i] = static_cast<float>(std::sin(static_cast<double>(i) * 3.141592653589793 * 2.0 / 65536.0));
    return t;
}
inline const std::array<float, 65536> SIN_TABLE = makeSinTable();
}

struct Mth {
    static float sin(float f) { return detail::SIN_TABLE[javaIntCast(f * 10430.378f) & 0xFFFF]; }
    static float cos(float f) { return detail::SIN_TABLE[javaIntCast(f * 10430.378f + 16384.0f) & 0xFFFF]; }
    static float sqrt(float f) { return static_cast<float>(std::sqrt(f)); }

    static int floor(float f) {
        int n = javaIntCast(f);
        return f < static_cast<float>(n) ? n - 1 : n;
    }
    static int floor(double d) {
        int n = javaIntCast(d);
        return d < static_cast<double>(n) ? n - 1 : n;
    }
    static int ceil(double d) {
        int n = javaIntCast(d);
        return d > static_cast<double>(n) ? n + 1 : n;
    }
    static int ceil(float f) {
        int n = javaIntCast(f);
        return f > static_cast<float>(n) ? n + 1 : n;
    }

    static float clamp(float f, float lo, float hi) { return f < lo ? lo : (f > hi ? hi : f); }
    static double clamp(double d, double lo, double hi) { return d < lo ? lo : (d > hi ? hi : d); }
    static int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

    static double absMax(double d, double d2) {
        if (d < 0.0) d = -d;
        if (d2 < 0.0) d2 = -d2;
        return d > d2 ? d : d2;
    }

    static bool equal(float f, float f2) { return std::abs(f2 - f) < 1.0E-5f; }
    static bool equal(double d, double d2) { return std::abs(d2 - d) < (double)1.0E-5f; }

    static float wrapDegrees(float f) {
        float f2 = std::fmod(f, 360.0f);
        if (f2 >= 180.0f) f2 -= 360.0f;
        if (f2 < -180.0f) f2 += 360.0f;
        return f2;
    }
    static double wrapDegrees(double d) {
        double d2 = std::fmod(d, 360.0);
        if (d2 >= 180.0) d2 -= 360.0;
        if (d2 < -180.0) d2 += 360.0;
        return d2;
    }

    static double lengthSquared(double d, double d2, double d3) { return d * d + d2 * d2 + d3 * d3; }
    static double square(double d) { return d * d; }
    static float square(float f) { return f * f; }

    static double lerp(double d, double d2, double d3) { return d2 + d * (d3 - d2); }
    static double rotLerp(double d, double d2, double d3) {
        return d2 + d * wrapDegrees(d3 - d2);
    }

    static int8_t packDegrees(float f) { return (int8_t)(uint8_t)(floor(f * 256.0f / 360.0f) & 0xFF); }
    static float unpackDegrees(int8_t by) { return (float)((int)by * 360) / 256.0f; }

    static int sign(double d) { return d == 0.0 ? 0 : (d > 0.0 ? 1 : -1); }
};

}
