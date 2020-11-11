#include "catboost.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "../src/json.hpp"

namespace {
std::string root_path = ".";

std::string path_to(const std::string& filename) { return root_path + "/" + filename; }

} // namespace

#define CHECK(some)                                                \
    do {                                                           \
        if (!(some)) {                                             \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "]" \
                      << "Error: " << #some << " failed.\n";       \
            std::exit(1);                                          \
        } else {                                                   \
            std::cout << "[" << __FILE__ << ":" << __LINE__ << "]" \
                      << "Info: " << #some << " success\n";        \
        }                                                          \
    } while (0)

#define CHECK_FEQ(a, b, epsilon)                                                                              \
    do {                                                                                                      \
        double test_a = (a);                                                                                  \
        double test_b = (b);                                                                                  \
        double test_delta = test_a - test_b;                                                                  \
        if (test_delta < 0.0) test_delta = -test_delta;                                                       \
        if (test_delta > epsilon) {                                                                           \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "]"                                            \
                      << "Error: " << #a << "(" << test_a << ") != " #b << "(" << test_b << ")" << std::endl; \
            std::exit(1);                                                                                     \
        }                                                                                                     \
    } while (0)

struct Test {
    std::vector<std::vector<float>> x;
    std::vector<float> y;
};

static bool one_test(const std::string& name) {
    Test data;
    {
        std::ifstream f{path_to("testdata/" + name + ".json")};
        nlohmann::json value = nlohmann::json::parse(f);
        for (const auto& x : value.at("x")) {
            std::vector<float> v;
            for (const auto& a : x) {
                v.push_back(a.get<double>());
            }
            data.x.push_back(v);
        }

        for (const auto& y : value.at("y")) {
            data.y.push_back(y.get<double>());
        }
    }

    catboost::Model model;
    model.load(path_to("testdata/" + name + "-model.json"));
    for (size_t i = 0; i < data.x.size() && i < data.y.size(); ++i) {
        float p = model.apply(data.x[i]);
        CHECK_FEQ(p, data.y[i], 0.001);
    }

    std::vector<double> y;
    model.apply(data.x, y);

    CHECK(data.x.size() == y.size());
    for (size_t i = 0; i < data.x.size() && i < data.y.size(); ++i) {
        CHECK_FEQ(y[i], data.y[i], 0.001);
    }

    return true;
}

void test_catboost() {
    CHECK(one_test("xor"));
    CHECK(one_test("or"));
    CHECK(one_test("and"));
    CHECK(one_test("regression"));
}

int main(int argc, char** argv) {
#define ARG_FLAG(f, var)                   \
    if (!std::strcmp(argv[argindex], f)) { \
        var = true;                        \
        ++argindex;                        \
        continue;                          \
    }

#define ARG_STR(f, var)                                                                  \
    if (!std::strcmp(argv[argindex], f)) {                                               \
        if (argindex + 1 == argc) {                                                      \
            std::cerr << "Error: no parameter for argument '" << f << "'." << std::endl; \
            return 1;                                                                    \
        }                                                                                \
        var = std::string(argv[++argindex]);                                             \
        ++argindex;                                                                      \
        continue;                                                                        \
    }

    int argindex = 1;
    bool help = false;
    while (argindex < argc) {
        if (help) {
            std::cout << "Usage: unittest [-d root_path]" << std::endl;
            std::cout << "root_path is a path to test_data." << std::endl;
            return 0;
        }
        ARG_STR("-d", root_path)
        ARG_STR("--test-data", root_path)
        ARG_FLAG("-h", help);
        ARG_FLAG("--help", help);

        std::cerr << "Error: invalid command line argument '" << argv[argindex] << "'" << std::endl;
        return 1;
    }

    test_catboost();

    return 0;
}
