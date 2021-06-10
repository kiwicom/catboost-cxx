#include <cb.h>
#include <catboost.hpp>
#include <limits>
#include <sstream>

// Implementation of the C interface

struct catboost_model_info_st {
    catboost::Model model;
};

static thread_local std::string cb_last_error;

#define CB_BEGIN try
#define CB_END(rv) \
    catch (const std::exception& exc) { \
        cb_last_error = exc.what(); \
        return rv; \
    } catch (...) { \
        cb_last_error = "Unknown error"; \
        return rv; \
    }

extern "C" catboost_model_info_t* cb_model_load(const char* filename) {
    CB_BEGIN {
        auto model = std::make_unique<catboost_model_info_t>();
        model->model.load(std::string(filename));
        return model.release();
    } CB_END(nullptr)
}

extern "C" catboost_model_info_t* cb_model_load_from_string(const char* data, size_t data_len) {
    // TODO: do not copy memory
    CB_BEGIN {
        std::string buf{data, data_len};
        std::istringstream ss{buf};
        auto model = std::make_unique<catboost_model_info_t>();
        model->model.load(ss);
        return model.release();
    } CB_END(nullptr)
}

extern "C" void cb_model_free(catboost_model_info_t* model) {
    CB_BEGIN {
        delete model;
    } CB_END()
}

extern "C" double cb_model_apply(const catboost_model_info_t* model, const float* features, size_t count) {
    CB_BEGIN {
        return model->model.apply(features, count);
    } CB_END(std::numeric_limits<double>::quiet_NaN())
}

extern "C" int cb_model_apply_many(const catboost_model_info_t* model, const float* const* features, size_t size, size_t count, double* y) {
    CB_BEGIN {
        model->model.apply(features, size, count, y);
        return 0;
    } CB_END(-1);
}

extern "C" size_t cb_model_feature_count(const catboost_model_info_t* model) {
    CB_BEGIN {
        return model->model.feature_count();
    } CB_END(0)
}

const char* cb_model_last_error(void) {
    return cb_last_error.c_str();
}

void cb_model_last_error_clear(void) {
    cb_last_error = "";
}

