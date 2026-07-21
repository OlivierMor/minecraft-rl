#include "runtime/logger.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace rl {

void CsvLogger::log(const std::vector<std::pair<std::string, double>>& row) {
    bool fresh = !std::filesystem::exists(path_);
    if (columns_.empty()) {
        if (!fresh) {
            std::ifstream in(path_);
            std::string header;
            std::getline(in, header);
            std::stringstream ss(header);
            std::string col;
            while (std::getline(ss, col, ',')) columns_.push_back(col);
        }
        if (columns_.empty()) {
            for (const auto& [k, v] : row) columns_.push_back(k);
        } else {
            std::vector<std::string> added;
            for (const auto& [k, v] : row) {
                bool known = false;
                for (const auto& c : columns_)
                    if (c == k) { known = true; break; }
                if (!known) added.push_back(k);
            }
            if (!added.empty()) {
                std::ifstream in(path_);
                std::string line;
                std::vector<std::string> lines;
                while (std::getline(in, line)) lines.push_back(line);
                in.close();
                std::string pad(added.size(), ',');
                for (const auto& k : added) columns_.push_back(k);
                std::ofstream out(path_, std::ios::trunc);
                for (size_t i = 0; i < columns_.size(); ++i)
                    out << columns_[i] << (i + 1 < columns_.size() ? "," : "\n");
                for (size_t i = 1; i < lines.size(); ++i) out << lines[i] << pad << "\n";
            }
        }
    }
    std::ofstream f(path_, std::ios::app);
    if (fresh) {
        for (size_t i = 0; i < columns_.size(); ++i)
            f << columns_[i] << (i + 1 < columns_.size() ? "," : "\n");
    }
    for (size_t i = 0; i < columns_.size(); ++i) {
        bool found = false;
        for (const auto& [k, v] : row)
            if (k == columns_[i]) {
                f << v;
                found = true;
                break;
            }
        (void)found;
        f << (i + 1 < columns_.size() ? "," : "\n");
    }
}

}
