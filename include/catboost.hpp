#pragma once

#include <string>
#include <istream>
#include <vector>
#include <memory>

/**
 * @brief Minimal CatBoost model applier in C++
 */

namespace catboost {

	class Model {
		struct Impl;
		std::unique_ptr<Impl> impl_;
        double bias_ = 0.0;
        double scale_ = 1.0;
	public:
		Model(const Model&) = delete;
		Model(Model&&) = delete;
		Model& operator = (const Model&) = delete;
		Model& operator = (Model&&) = delete;

		/// Create empty model.
		Model();

		/// Load model from file.
		/// @argument filename - name of file to load model from.
		/// Model should be stored in JSON format.
		explicit Model(const std::string& filename);

		/// Load model from file.
		/// @argument in - stream to load model from.
		/// Model should be stored in JSON format.
		explicit Model(std::istream& in);

		~Model();

		/// Load model from file.
		/// @argument filename - name of file to load model from.
		/// Model should be stored in JSON format.
		void load(const std::string& filename);

		/// Load model from file.
		/// @argument in - stream to load model from.
		/// Model should be stored in JSON format.
		void load(std::istream& in);

		/// Apply model to features.
		/// @argument features - pointer to array of features
		/// @argument count - number of factors provided
		/// @returns predicted value
		double apply(const float* features, size_t count) const;

        /// Apply model to a bucket of examples.
        /// @argument features - array of arrays of features
        /// @argument size - number of examples in the set
        /// @argument count - number of features for each example
        /// @argument y - array to save predicted values.
        /// This function is equal to:
        /// for (size_t i = 0; i < size; ++i)
        ///     y[i] = predict([features[i]], count);
        /// but more efficient because of vectorization.
        void apply(const float* const* features, size_t size, size_t count, double* y) const;

		/// Apply model to features.
		/// @argument features - vector of features
		/// @returns predicted value
		double apply(const std::vector<float>& features) const {
			return apply(features.data(), features.size());
		}

        /// Apply model to a bucket of examples.
        /// Each example should have not less features than feature count.
        /// @argument features - input vectors
        /// @argument y - output predictions. This vector will be resized to the correct size automatically.
		void apply(const std::vector<std::vector<float>>& features, std::vector<double>& y) const {
            static constexpr size_t max_bucket = 16;
            const float* bucket[max_bucket];
            const size_t fcount = feature_count();

            if (features.empty()) {
                return;
            }

            y.resize(features.size());
            size_t i = 0;

            // Process examples using buckets of size 16:
            for (i = 0; i + max_bucket <= features.size(); i += max_bucket) {
                for (size_t j = 0; j < max_bucket; ++j) {
                    if (features[i + j].size() < fcount) {
                        throw std::runtime_error("Not enought features for model");
                    }

                    bucket[j] = features[i + j].data();
                }
                apply(bucket, max_bucket, fcount, y.data() + i);
            }

            size_t cnt = 0;
            // Process rest of examples one by one:
            for (; i + cnt < features.size(); ++cnt) {
                if (features[i + cnt].size() < fcount) {
                    throw std::runtime_error("Not enought features for model");
                }
                bucket[cnt] = features[i + cnt].data();
            }
            apply(bucket, cnt, fcount, y.data() + i);
		}

        /// Return number of features model was trainer on.
        size_t feature_count() const;
	};

} // namespace catboost

