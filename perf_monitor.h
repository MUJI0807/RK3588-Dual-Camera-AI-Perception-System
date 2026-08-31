#ifndef PERF_MONITOR_H
#define PERF_MONITOR_H

#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <cstdint>

struct PerfSample {
    int64_t timestamp_ms;
    float cpu_total;
    std::vector<float> cpu_cores;

    float npu_core[3];
    bool npu_available;
};

class PerfMonitor {
public:
    static PerfMonitor& instance();

    void start(const std::string& log_path = "perf_log.csv", int interval_ms = 500);
    void stop();

    ~PerfMonitor();

    PerfMonitor(const PerfMonitor&) = delete;
    PerfMonitor& operator=(const PerfMonitor&) = delete;

private:
    PerfMonitor() = default;

    struct CpuStat {
        uint64_t user, nice, system, idle, iowait, irq, softirq, steal;

        uint64_t total() const {
            return user + nice + system + idle + iowait + irq + softirq + steal;
        }

        uint64_t active() const {
            return user + nice + system + irq + softirq + steal;
        }
    };

    void monitor_loop();
    bool read_cpu_stats(std::vector<CpuStat>& out);
    float calc_cpu_usage(const CpuStat& prev, const CpuStat& curr);
    bool read_npu_load(float core[3]);
    void flush_log();

private:
    std::string log_path_;
    int interval_ms_ = 500;

    std::atomic<bool> running_{false};
    std::thread monitor_thread_;

    std::vector<PerfSample> samples_;
    int64_t start_time_ms_ = 0;
};

#endif