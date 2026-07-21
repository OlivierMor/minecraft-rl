#include "config.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rl {

namespace {

struct Parser {
    const std::string& t;
    size_t i = 0;
    int line = 1;

    explicit Parser(const std::string& text) : t(text) {}

    [[noreturn]] void fail(const std::string& msg) {
        throw std::runtime_error("config parse error, line " + std::to_string(line) + ": " + msg);
    }
    bool eof() const { return i >= t.size(); }
    char peek() const { return t[i]; }
    void skipWs(bool newlines) {
        while (!eof()) {
            char c = t[i];
            if (c == '#') { while (!eof() && t[i] != '\n') ++i; continue; }
            if (c == '\n') {
                if (!newlines) return;
                ++line; ++i; continue;
            }
            if (c == ' ' || c == '\t' || c == '\r') { ++i; continue; }
            return;
        }
    }

    std::string ident() {
        size_t s = i;
        while (!eof() && (std::isalnum((unsigned char)t[i]) || t[i] == '_' || t[i] == '.' || t[i] == '-'))
            ++i;
        if (s == i) fail("expected identifier");
        return t.substr(s, i - s);
    }

    Value value() {
        skipWs(false);
        if (eof()) fail("expected value");
        char c = peek();
        Value v;
        if (c == '"') {
            ++i;
            v.kind = Value::Str;
            while (!eof() && t[i] != '"') {
                if (t[i] == '\n') fail("newline in string");
                if (t[i] == '\\' && i + 1 < t.size()) ++i;
                v.s += t[i++];
            }
            if (eof()) fail("unterminated string");
            ++i;
            return v;
        }
        if (c == '[') {
            ++i;
            v.kind = Value::Arr;
            skipWs(true);
            while (!eof() && peek() != ']') {
                v.a.push_back(value());
                skipWs(true);
                if (!eof() && peek() == ',') { ++i; skipWs(true); }
            }
            if (eof()) fail("unterminated array");
            ++i;
            return v;
        }
        if (std::isalpha((unsigned char)c)) {
            std::string w = ident();
            if (w == "true")  { v.kind = Value::Bool; v.b = true;  return v; }
            if (w == "false") { v.kind = Value::Bool; v.b = false; return v; }
            fail("unknown literal '" + w + "'");
        }
        size_t s = i;
        while (!eof() && (std::isdigit((unsigned char)t[i]) || t[i] == '+' || t[i] == '-' ||
                          t[i] == '.' || t[i] == 'e' || t[i] == 'E' || t[i] == '_'))
            ++i;
        if (s == i) fail("expected value");
        std::string w = t.substr(s, i - s);
        std::string clean;
        for (char ch : w) if (ch != '_') clean += ch;
        char* end = nullptr;
        v.kind = Value::Num;
        v.n = std::strtod(clean.c_str(), &end);
        if (end == nullptr || *end != '\0') fail("bad number '" + w + "'");
        return v;
    }

    void parse(std::map<std::string, Value>& out) {
        std::string section;
        while (true) {
            skipWs(true);
            if (eof()) return;
            if (peek() == '[') {
                ++i;
                skipWs(false);
                section = ident();
                skipWs(false);
                if (eof() || peek() != ']') fail("expected ']'");
                ++i;
                continue;
            }
            std::string key = ident();
            skipWs(false);
            if (eof() || peek() != '=') fail("expected '=' after key '" + key + "'");
            ++i;
            Value v = value();
            std::string full = section.empty() ? key : section + "." + key;
            out[full] = std::move(v);
            skipWs(false);
            if (!eof() && peek() != '\n') fail("trailing junk after value for '" + full + "'");
        }
    }
};

}

Config Config::fromString(const std::string& text) {
    Config c;
    c.raw_ = text;
    Parser p(text);
    p.parse(c.kv_);
    return c;
}

Config Config::fromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return fromString(ss.str());
}

std::string Config::str(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    used_.insert(key);
    if (it->second.kind != Value::Str)
        throw std::runtime_error("config key '" + key + "' is not a string");
    return it->second.s;
}

double Config::num(const std::string& key, double def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    used_.insert(key);
    if (it->second.kind != Value::Num)
        throw std::runtime_error("config key '" + key + "' is not a number");
    return it->second.n;
}

bool Config::boolean(const std::string& key, bool def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    used_.insert(key);
    if (it->second.kind != Value::Bool)
        throw std::runtime_error("config key '" + key + "' is not a bool");
    return it->second.b;
}

std::vector<Value> Config::arr(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return {};
    used_.insert(key);
    if (it->second.kind != Value::Arr)
        throw std::runtime_error("config key '" + key + "' is not an array");
    return it->second.a;
}

std::vector<double> Config::numArr(const std::string& key) const {
    std::vector<double> out;
    for (const Value& v : arr(key)) {
        if (v.kind != Value::Num)
            throw std::runtime_error("config key '" + key + "': expected numeric array");
        out.push_back(v.n);
    }
    return out;
}

std::vector<std::string> Config::strArr(const std::string& key) const {
    std::vector<std::string> out;
    for (const Value& v : arr(key)) {
        if (v.kind != Value::Str)
            throw std::runtime_error("config key '" + key + "': expected string array");
        out.push_back(v.s);
    }
    return out;
}

double Config::schedule(const std::string& key, double x, double def) const {
    auto pts = arr(key);
    if (pts.empty()) return def;
    double px = 0, py = def;
    bool first = true;
    for (const Value& p : pts) {
        if (p.kind != Value::Arr || p.a.size() != 2 || p.a[0].kind != Value::Num || p.a[1].kind != Value::Num)
            throw std::runtime_error("config key '" + key + "': expected [[x,y],...] schedule");
        double cx = p.a[0].n, cy = p.a[1].n;
        if (x <= cx) {
            if (first) return cy;
            double t = (x - px) / (cx - px);
            return py + (cy - py) * t;
        }
        px = cx; py = cy; first = false;
    }
    return py;
}

std::vector<std::string> Config::unusedKeys() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : kv_)
        if (!used_.count(k)) out.push_back(k);
    return out;
}

}
