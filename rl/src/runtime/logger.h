#pragma once
#include <string>
#include <utility>
#include <vector>

namespace rl {

class CsvLogger {
public:
    explicit CsvLogger(std::string path) : path_(std::move(path)) {}
    void log(const std::vector<std::pair<std::string, double>>& row);

private:
    std::string path_;
    std::vector<std::string> columns_;
};

}
