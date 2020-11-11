#include <unistd.h>

#include <cstring>
#include <functional>
#include <set>

#include "catboost.hpp"
#include "codrna.cpp"
#include "creditgermany.cpp"
#include "dll.hpp"
#include "msrank.cpp"
#include "perf_test.hpp"

#ifdef WIN32
#define CATBOOST_LIBRARY "build/libcatboostmodel.dll"
#else
#define CATBOOST_LIBRARY "build/libcatboostmodel.so"
#endif

typedef void ModelCalcerHandle;

static struct CatboostAPIInitializer {
    CatboostAPIInitializer() {
        dll_ = std::make_unique<DLL>(CATBOOST_LIBRARY);
        if (*dll_) {
            CalcModelPredictionSingle = dll_->sym("CalcModelPredictionSingle");
            CalcModelPrediction = dll_->sym("CalcModelPrediction");
            ModelCalcerCreate = dll_->sym("ModelCalcerCreate");
            ModelCalcerDelete = dll_->sym("ModelCalcerDelete");
            LoadFullModelFromFile = dll_->sym("LoadFullModelFromFile");

            if (!CalcModelPredictionSingle || !ModelCalcerCreate || !ModelCalcerDelete || !LoadFullModelFromFile ||
                !CalcModelPrediction) {
                std::cerr << "Can not load catboost libray!" << std::endl;
                dll_.reset(nullptr);
            }
        } else {
            dll_.reset(nullptr);
        }
    }

    std::unique_ptr<DLL> dll_;

    bool (*CalcModelPredictionSingle)(ModelCalcerHandle* modelHandle, const float* floatFeatures,
                                      size_t floatFeaturesSize, const char** catFeatures, size_t catFeaturesSize,
                                      double* result, size_t resultSize) = nullptr;
    ModelCalcerHandle* (*ModelCalcerCreate)() = nullptr;
    void (*ModelCalcerDelete)(ModelCalcerHandle* modelHandle) = nullptr;
    bool (*LoadFullModelFromFile)(ModelCalcerHandle*, const char*) = nullptr;
    bool (*CalcModelPrediction)(ModelCalcerHandle* modelHandle, size_t docCount, const float** floatFeatures,
                                size_t floatFeaturesSize, const char*** catFeatures, size_t catFeaturesSize,
                                double* result, size_t resultSize);

    operator bool() const { return dll_.get() != nullptr; }
} CatboostAPI;

struct JsonModel {
    catboost::Model model_;

    explicit JsonModel(const std::string& filename) { model_.load(filename); }

    double predict(const std::vector<float>& x) const { return model_.apply(x); }

    void predict(const std::vector<std::vector<float>>& x, std::vector<double>& y) const { model_.apply(x, y); }
};

struct YaModel {
    ModelCalcerHandle* handle_ = nullptr;

    YaModel(const std::string& filename) {
        if (!CatboostAPI) {
            return;
        }

        handle_ = CatboostAPI.ModelCalcerCreate();
        if (!handle_) {
            throw std::runtime_error("Can't create model");
        }

        if (!CatboostAPI.LoadFullModelFromFile(handle_, filename.c_str())) {
            throw std::runtime_error("Can't load model");
        }
    }

    ~YaModel() {
        if (handle_) {
            CatboostAPI.ModelCalcerDelete(handle_);
        }
    }

    double predict(const std::vector<float>& x) const {
        double res;
        if (!CatboostAPI.CalcModelPredictionSingle(handle_, x.data(), x.size(), // floatFeatures
                                                   nullptr, 0,                  // catFeatures
                                                   &res, 1)) {
            throw std::runtime_error("Prediction failed");
        }

        return res;
    }

    void predict(const std::vector<std::vector<float>>& x, std::vector<double>& y) const {
        const float* buf[32];
        size_t count = x[0].size();
        y.resize(x.size());
        size_t i = 0;
        for (i = 0; i + 32 <= x.size(); i += 32) {
            for (size_t j = 0; j < 32; ++j) {
                buf[j] = x[i + j].data();
            }
            if (!CatboostAPI.CalcModelPrediction(handle_, 32, buf, count, nullptr, 0, y.data() + i, 32)) {
                throw std::runtime_error("Prediction failed");
            }
        }

        size_t cnt = 0;
        for (; i + cnt < x.size(); ++cnt) {
            buf[cnt] = x[i + cnt].data();
        }

        if (!CatboostAPI.CalcModelPrediction(handle_, cnt, buf, count, nullptr, 0, y.data() + i, cnt)) {
            throw std::runtime_error("Prediction failed");
        }
    }

    operator bool() const { return handle_ != nullptr; }
};

template <typename SModel>
struct SingleTest {
    std::string name;
    TestData data;
    SModel smodel;
    JsonModel jmodel;
    YaModel ymodel;
    bool do_not_run_static = false;
    bool do_not_run_yandex = !CatboostAPI;
    bool do_not_run_compare = false;

    SingleTest(const std::string& base_name)
        : name{base_name}, jmodel{base_name + ".json"}, ymodel{base_name + ".cbm"} {
        data.load_tsv(base_name + "_test.tsv");
    }

    void perf_tests() {
        if (!do_not_run_static) {
            std::cout << name << ": static compiled model" << std::endl;
            perf_test(smodel, data, 5);
        }

        std::cout << name << ": this library" << std::endl;
        perf_test(jmodel, data, 5);

        if (!do_not_run_yandex) {
            if (ymodel) {
                std::cout << name << ": Yandex library" << std::endl;
                perf_test(ymodel, data, 5);
            } else {
                std::cerr << "WARNING: do not test Catboost library because it was not loaded!" << std::endl;
            }
        }
    }

    void perf_bucket() {
        if (!do_not_run_static) {
            std::cout << name << ": bucket static compiled model" << std::endl;
            perf_test_buckets(smodel, data, 5);
        }

        std::cout << name << ": bucket this library" << std::endl;
        perf_test_buckets(jmodel, data, 5);

        if (!do_not_run_yandex) {
            if (ymodel) {
                std::cout << name << ": bucket Yandex library" << std::endl;
                perf_test_buckets(ymodel, data, 5);
            } else {
                std::cerr << "WARNING: do not test Catboost library because it was not loaded!" << std::endl;
            }
        }
    }

    void compare() {
        double sum = 0.0;
        double sum2 = 0.0;
        std::vector<double> bucket;
        bucket.reserve(data.data.size());

        jmodel.predict(data.data, bucket);

        for (size_t i = 0; i < data.data.size(); ++i) {
            const auto& x = data.data[i];
            double etalon = smodel.predict(x);
            double y = jmodel.predict(x);

            double delta = std::abs(y - etalon);
            double delta2 = std::abs(bucket[i] - etalon);
            sum += delta;
            sum2 += delta2;

            if (delta > 1e-5) {
                std::cerr << "WARNING: Delta is too big (" << delta << ") for line " << i << std::endl;
            }

            if (delta2 > 1e-5) {
                std::cerr << "WARNING: Delta for bucket calc is too big (" << delta2 << ") for line " << i << std::endl;
            }
        }

        std::cout << "Average delta: " << (sum / data.data.size()) << std::endl;
        std::cout << "Average delta bucket: " << (sum2 / data.data.size()) << std::endl;
    }

    void run() {
        if (!do_not_run_compare) {
            compare();
        }
        perf_tests();
        perf_bucket();
    }
};

class CmdLine {
    struct HelpMessage {
        std::vector<std::string> names;
        std::string help;
        bool flag = true;
        std::function<void(const char*)> action;

        bool match(const char* s) const {
            for (const auto& nm : names) {
                if (nm == s) {
                    return true;
                }
            }
            return false;
        }
    };
    std::vector<HelpMessage> args;
    std::string description;

public:
    explicit CmdLine(const std::string& descr) : description(descr) {}

    void print_usage(const std::string& prog) const {
        std::cout << "Usage:" << std::endl;
        std::cout << prog;
        for (const auto& msg : args) {
            std::cout << " [" << msg.names[0];
            if (!msg.flag) {
                std::cout << " <val>";
            }
            std::cout << "]";
        }
        std::cout << std::endl;
        std::cout << "    " << description << std::endl;

        std::cout << "Arguments:" << std::endl;
        for (const auto& msg : args) {
            for (size_t i = 0; i < msg.names.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << msg.names[i];
            }
            if (!msg.flag) std::cout << " <value>";
            std::cout << std::endl;
            std::cout << "    " << msg.help << std::endl;
        }
    }

    CmdLine& flag(const std::string& f, bool& var, const std::string& help) {
        args.emplace_back();
        args.back().names.push_back(f);
        args.back().flag = true;
        args.back().help = help;
        args.back().action = [&var](const char* v) { var = (v[0] == '1'); };

        return *this;
    }

    CmdLine& synonym(const std::string& f) {
        args.back().names.push_back(f);
        return *this;
    }

    // synonym, but better :)
    CmdLine& aka(const std::string& f) {
        args.back().names.push_back(f);
        return *this;
    }

    CmdLine& arg(const std::string& f, std::string& var, const std::string& help) {
        args.emplace_back();
        args.back().names.push_back(f);
        args.back().flag = false;
        args.back().help = help;
        args.back().action = [&var](const char* v) { var = v; };

        return *this;
    }

    template <typename F>
    CmdLine& action(const std::string& f, F&& func, const std::string& help) {
        args.emplace_back();
        args.back().names.push_back(f);
        args.back().flag = false;
        args.back().help = help;
        args.back().action = std::forward<F>(func);

        return *this;
    }

    bool parse(int argc, const char* argv[]) {
        int argidx = 1;
        while (argidx < argc) {
            if (!std::strcmp(argv[argidx], "-h") || !std::strcmp(argv[argidx], "--help")) {
                print_usage(argv[0]);
                return false;
            }

            bool matched = false;
            for (const auto& arg : args) {
                if (arg.match(argv[argidx])) {
                    matched = true;
                    if (arg.flag) {
                        arg.action("1");
                    } else {
                        if (argidx + 1 < argc) {
                            arg.action(argv[++argidx]);
                        } else {
                            std::cerr << "Error: option " << argv[argidx] << " needs argument!" << std::endl;
                            return false;
                        }
                    }
                    ++argidx;
                }

                if (matched) break;
            }

            if (!matched) {
                std::cerr << "Error: unknown argument: " << argv[argidx] << std::endl;
                print_usage(argv[0]);
                return false;
            }
        }

        return true;
    }
};

int main(int argc, const char* argv[]) {
    std::set<std::string> list_tests;
    std::string root_path;
    bool do_not_run_static = false;
    bool do_not_run_yandex = false;
    bool do_not_run_compare = false;

    CmdLine args{"run performance tests."};
    args.arg("-d", root_path, "path to the tests directory (default: .)")
        .aka("--test-data")
        .action(
            "-t", [&list_tests](const char* v) { list_tests.emplace(v); },
            "run this test. Default is to run all tests.")
        .aka("--run-tests")
        .flag("--no-static", do_not_run_static, "do not run static model tests")
        .flag("--no-yandex", do_not_run_yandex, "do not run Yandex model library tests")
        .flag("--no-compare", do_not_run_compare, "run only performance test, no values comparation");

    if (!args.parse(argc, argv)) {
        return 1;
    }

    if (!root_path.empty()) {
        chdir(root_path.c_str());
    }

    if (list_tests.empty()) {
        list_tests.emplace("msrank");
        list_tests.emplace("creditgermany");
        list_tests.emplace("codrna");
    }

    if (list_tests.count("msrank")) {
        SingleTest<StaticMSRankModel> model{"msrank"};
        model.do_not_run_static = do_not_run_static;
        model.do_not_run_yandex = do_not_run_yandex;
        model.do_not_run_compare = do_not_run_compare;
        model.run();
    }

    if (list_tests.count("creditgermany")) {
        SingleTest<StaticCreditGermanyModel> model{"creditgermany"};
        model.do_not_run_static = do_not_run_static;
        model.do_not_run_yandex = do_not_run_yandex;
        model.do_not_run_compare = do_not_run_compare;
        model.run();
    }

    if (list_tests.count("codrna")) {
        SingleTest<StaticCodRNAModel> model{"codrna"};
        model.do_not_run_static = do_not_run_static;
        model.do_not_run_yandex = do_not_run_yandex;
        model.do_not_run_compare = do_not_run_compare;
        model.run();
    }

    return 0;
}
