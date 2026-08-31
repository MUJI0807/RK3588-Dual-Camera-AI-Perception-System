#include "perf_monitor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>

PerfMonitor& PerfMonitor::instance()
{
    static PerfMonitor monitor;
    return monitor;
}

PerfMonitor::~PerfMonitor()
{
    stop();
}

void PerfMonitor::start(const std::string& log_path, int interval_ms)
{
    if (running_) {
        return;
    }

    log_path_ = log_path;
    interval_ms_ = interval_ms;

    samples_.clear();
    samples_.reserve(4096);

    start_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();

    running_ = true;
    monitor_thread_ = std::thread(&PerfMonitor::monitor_loop, this);

    std::cout << "[PerfMonitor] start, log path: "
              << log_path_ << ", interval: "
              << interval_ms_ << " ms" << std::endl;
}

void PerfMonitor::stop()
{
    if (!running_) {
        return;
    }

    running_ = false;

    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }

    flush_log();

    std::cout << "[PerfMonitor] stop, samples: "
              << samples_.size()
              << ", saved to: " << log_path_
              << std::endl;
}

void PerfMonitor::monitor_loop()
{
    std::vector<CpuStat> prev_stats;
    read_cpu_stats(prev_stats);

    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));

        std::vector<CpuStat> curr_stats;
        if (!read_cpu_stats(curr_stats)) {
            continue;
        }

        PerfSample sample{};
        sample.timestamp_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count() - start_time_ms_;

        sample.cpu_total = 0.0f;

        if (!prev_stats.empty() && !curr_stats.empty()) {
            sample.cpu_total = calc_cpu_usage(prev_stats[0], curr_stats[0]);

            size_t core_count = std::min(prev_stats.size(), curr_stats.size());
            for (size_t i = 1; i < core_count; ++i) {
                sample.cpu_cores.push_back(
                    calc_cpu_usage(prev_stats[i], curr_stats[i])
                );
            }
        }

        prev_stats = curr_stats;

        sample.npu_available = read_npu_load(sample.npu_core);

        samples_.push_back(sample);
    }
}

bool PerfMonitor::read_cpu_stats(std::vector<CpuStat>& out)
{
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        return false;
    }

    out.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 3) != "cpu") {
            break;
        }

        if (line.size() < 4) {
            continue;
        }

        if (line[3] != ' ' && (line[3] < '0' || line[3] > '9')) {
            continue;
        }

        std::istringstream iss(line);
        std::string label;
        CpuStat stat{};

        iss >> label
            >> stat.user
            >> stat.nice
            >> stat.system
            >> stat.idle
            >> stat.iowait
            >> stat.irq
            >> stat.softirq
            >> stat.steal;

        out.push_back(stat);
    }

    return !out.empty();
}

float PerfMonitor::calc_cpu_usage(const CpuStat& prev, const CpuStat& curr)
{
    uint64_t total_delta = curr.total() - prev.total();
    uint64_t active_delta = curr.active() - prev.active();

    if (total_delta == 0) {
        return 0.0f;
    }

    return static_cast<float>(active_delta) * 100.0f /
           static_cast<float>(total_delta);
}

bool PerfMonitor::read_npu_load(float core[3])
{
    const char* path = "/sys/kernel/debug/rknpu/load";

    core[0] = -1.0f;
    core[1] = -1.0f;
    core[2] = -1.0f;

    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::getline(file, line);

    for (int i = 0; i < 3; ++i) {
        std::string key = "Core" + std::to_string(i) + ":";

        size_t pos = line.find(key);
        if (pos == std::string::npos) {
            continue;
        }

        pos += key.size();

        while (pos < line.size() && line[pos] == ' ') {
            ++pos;
        }

        size_t end = line.find('%', pos);
        if (end == std::string::npos) {
            continue;
        }

        try {
            core[i] = std::stof(line.substr(pos, end - pos));
        } catch (...) {
            core[i] = -1.0f;
        }
    }

    return true;
}

void PerfMonitor::flush_log()
{
    if (samples_.empty()) {
        return;
    }

    std::ofstream file(log_path_);
    if (!file.is_open()) {
        std::cerr << "[PerfMonitor] failed to open log file: "
                  << log_path_ << std::endl;
        return;
    }

    size_t max_cpu_cores = 0;
    for (const auto& sample : samples_) {
        max_cpu_cores = std::max(max_cpu_cores, sample.cpu_cores.size());
    }

    file << "timestamp_ms,cpu_total%";

    for (size_t i = 0; i < max_cpu_cores; ++i) {
        file << ",cpu" << i << "%";
    }

    file << ",npu_core0%,npu_core1%,npu_core2%,npu_available\n";

    file << std::fixed << std::setprecision(1);

    for (const auto& sample : samples_) {
        file << sample.timestamp_ms << ","
             << sample.cpu_total;

        for (size_t i = 0; i < max_cpu_cores; ++i) {
            file << ",";
            if (i < sample.cpu_cores.size()) {
                file << sample.cpu_cores[i];
            } else {
                file << "0.0";
            }
        }

        file << ","
             << sample.npu_core[0] << ","
             << sample.npu_core[1] << ","
             << sample.npu_core[2] << ","
             << (sample.npu_available ? 1 : 0)
             << "\n";
    }
}