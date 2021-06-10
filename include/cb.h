#ifndef CATBOOST_C_INTERFACE_H__INC
#define CATBOOST_C_INTERFACE_H__INC

#include <ctype.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct catboost_model_info_st catboost_model_info_t;

/// Load model from file.
/// @argument filename - name of file to load model from.
/// Returns loaded model. On error function returns NULL and sets reason string.
catboost_model_info_t* cb_model_load(const char* filename);

/// Load model from string representation.
/// @argument data - JSON model data
/// @argument data_len - size of the data
/// Returns loaded model. On error function returns NULL and sets reason string.
catboost_model_info_t* cb_model_load_from_string(const char* data, size_t data_len);

/// Free model memory.
/// @argument model - loaded model to free.
void cb_model_free(catboost_model_info_t* model);

/// Apply model to the list of features.
/// @argument model - loaded model to apply
/// @argument features - pointer to array of features
/// @argument count - number of factors provided
/// @returns predicted value. On error function returns NaN.
double cb_model_apply(const catboost_model_info_t* model, const float* features, size_t count);

/// Apply model to the bucket.
/// @argument model - loaded model to apply
/// @argument features - array of arrays of features
/// @argument size - number of examples in the set
/// @argument count - number of features for each example
/// @argument y - array to save predicted values.
/// This function is equal to:
/// for (size_t i = 0; i < size; ++i)
///     y[i] = cb_model_apply(model, [features[i]], count);
/// but more efficient because of vectorization.
/// @returns 0 on success, -1 on error.
int cb_model_apply_many(const catboost_model_info_t* model, const float* const* features, size_t size, size_t count, double* y);

/// Get number of features model was trained on.
/// @argument model - loaded model to apply
/// @returns number of features expected by the model.
size_t cb_model_feature_count(const catboost_model_info_t* model);

/// Get last error information as a string.
/// @returns last error description.
const char* cb_model_last_error(void);

/// Clear error message for this thread.
void cb_model_last_error_clear(void);

#ifdef __cplusplus
}
#endif

#endif

