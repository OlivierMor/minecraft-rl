#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace rl {

struct Value {
    enum Kind { Str, Num, Bool, Arr } kind = Num;
    std::string s;
    double n = 0;
    bool b = false;
    std::vector<Value> a;
};

class Config {
public:
    Config() = default;
    static Config fromFile(const std::string& path);
    static Config fromString(const std::string& text);

    std::string str(const std::string& key, const std::string& def = "") const;
    double num(const std::string& key, double def = 0) const;
    bool boolean(const std::string& key, bool def = false) const;
    std::vector<Value> arr(const std::string& key) const;
    std::vector<double> numArr(const std::string& key) const;
    std::vector<std::string> strArr(const std::string& key) const;
    double schedule(const std::string& key, double x, double def) const;

    const std::string& rawText() const { return raw_; }
    std::vector<std::string> unusedKeys() const;

private:
    std::map<std::string, Value> kv_;
    mutable std::set<std::string> used_;
    std::string raw_;
};

}
