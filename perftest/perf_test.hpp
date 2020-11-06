#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sys/time.h>

inline double ftime() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return double(tv.tv_sec) + double(tv.tv_usec) / 1000000.0;
}

struct TestData {
    std::vector<double> label;
    std::vector<std::vector<float>> data;

    std::vector<std::string> split_string(const std::string& s, const std::string& delimiter) {
        std::vector<std::string> res;
        size_t begin = 0;
        for (;;) {
            auto next = s.find(delimiter, begin);
            if (next == std::string::npos) {
                res.push_back(s.substr(begin, s.size() - begin));
                return res;
            }

            res.push_back(s.substr(begin, next - begin));
            begin = next + delimiter.size();
        }

        return res; // unreachable
    }

    double parse_double(const std::string& s) {
        char* end = nullptr;
        double res = std::strtod(s.data(), &end);
        if (!end || end == s.data()) {
            throw std::runtime_error("Can't parse float");
        }

        return res;
    }

    void load_tsv(const std::string& fnm) {
        std::string line;
        std::ifstream f{fnm};
        if (!f.is_open()) {
            throw std::runtime_error("Can't open file");
        }

        while (std::getline(f, line)) {
            // Parse line:
            auto sline = split_string(line.substr(0, line.size() - 1), "\t");
            if (sline.size() < 2) {
                throw std::runtime_error("Invalid dataset");
            }

            label.push_back(parse_double(sline[0]));
            std::vector<float> x;
            for (size_t i = 1; i < sline.size(); ++i) {
                x.push_back(parse_double(sline[i]));
            }
            data.emplace_back(std::move(x));
        }

        f.close();
    }

};

template <typename Model>
inline void perf_test(Model& model, const TestData& test_data, int iters = 1) {
    double begin = ftime();
    double best_time = 0.0;

    for (int iter = 0; iter < iters; ++iter) {
        std::cerr << "BEGIN ITERATION: " << (iter + 1) << std::endl;
        double iter_begin = ftime();
        double sum_deltas = 0.0;

        for (size_t i = 0; i < test_data.data.size(); ++i) {
            double y = model.predict(test_data.data[i]);
            double delta = y - test_data.label[i];
            if (delta < 0.0) delta = -delta;
            sum_deltas += delta;
        }

        double iter_end = ftime();
        double delta_time = iter_end - iter_begin;
        sum_deltas /= test_data.data.size();
        std::cerr << "END ITERATION: " << (iter + 1) << " (" << delta_time << " seconds)"
            << " Q = " << sum_deltas << std::endl;
        if (iter == 0 || best_time > delta_time) {
            best_time = delta_time;
        }
    }

    double end = ftime();
    double sum_time = end - begin;
    std::cerr << iters << " iterations have finished in " << (end - begin) << " seconds." << std::endl;
    std::cerr << "Best time is " << best_time << " (" << (best_time / test_data.data.size()) << " per prediction, " << (test_data.data.size() / best_time) << " predictions/sec)" << std::endl;
    std::cerr << "Average time is " << (sum_time / iters) <<  "(" << (sum_time / iters / test_data.data.size()) << " per prediction)" << std::endl;
}

template <typename Model>
inline void perf_test_buckets(Model& model, const TestData& test_data, int iters = 1) {
    double begin = ftime();
    double best_time = 0.0;

    for (int iter = 0; iter < iters; ++iter) {
        std::vector<double> y;
        y.reserve(test_data.data.size());
        std::cerr << "BEGIN ITERATION: " << (iter + 1) << std::endl;
        double iter_begin = ftime();
        double sum_deltas = 0.0;

        model.predict(test_data.data, y);
        for (size_t i = 0; i < test_data.data.size(); ++i) {
            double delta = y[i] - test_data.label[i];
            if (delta < 0.0) delta = -delta;
            sum_deltas += delta;
        }

        double iter_end = ftime();
        double delta_time = iter_end - iter_begin;
        sum_deltas /= test_data.data.size();
        std::cerr << "END ITERATION: " << (iter + 1) << " (" << delta_time << " seconds)"
            << " Q = " << sum_deltas << std::endl;
        if (iter == 0 || best_time > delta_time) {
            best_time = delta_time;
        }
    }

    double end = ftime();
    double sum_time = end - begin;
    std::cerr << iters << " iterations have finished in " << (end - begin) << " seconds." << std::endl;
    std::cerr << "Best time is " << best_time << " (" << (best_time / test_data.data.size()) << " per prediction, " << (test_data.data.size() / best_time) << " predictions/sec)" << std::endl;
    std::cerr << "Average time is " << (sum_time / iters) <<  "(" << (sum_time / iters / test_data.data.size()) << " per prediction)" << std::endl;
}

