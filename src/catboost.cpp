#include "catboost.hpp"
#include "json.hpp"
#include <fstream>

namespace catboost {

    namespace {

        struct Tree {
            std::vector<double> values;
            std::vector<float> borders;
            std::vector<uint32_t> indexes;

            Tree(const nlohmann::json& t) {
                const auto& splits = t.at("splits");
                const auto& values = t.at("leaf_values");

                if (splits.size() >= 32 || static_cast<size_t>(1) << splits.size() != values.size()) {
                    throw std::runtime_error("Invalid model");
                }

                // Loading values:
                this->values.resize(values.size());
                for (size_t i = 0; i < values.size(); ++i)
                    this->values[i] = values[i].get<double>();

                // Loading splits:
                for (const auto& split : splits) {
                    borders.push_back(split.at("border").get<double>());
                    indexes.push_back(split.at("float_feature_index").get<unsigned>());
                }
            }

            // Calculate raw result for one feature
            inline double calc(const float* f) const noexcept;
            // Caclulate raw result for 4 examples
            inline void calc4(const float* f0, const float* f1, const float* f2, const float* f3, double *res) const noexcept;

            size_t depth() const {
                return borders.size();
            }
        };

    } // anonymous namespace
// #define NOSSE
#ifdef NOSSE

    inline double Tree::calc(const float* f) const noexcept {
        size_t idx = 0;
        size_t one = 1;

        for (size_t i = 0; i < borders.size(); ++i) {
            if (f[indexes[i]] > borders[i]) {
                idx |= one;
            }
            one <<= 1;
        }

        return values[idx];
    }

    inline void Tree::calc4(const float* f0, const float* f1, const float* f2, const float* f3, double *res) const noexcept {
        res[0] += calc(f0);
        res[1] += calc(f1);
        res[2] += calc(f2);
        res[3] += calc(f3);
    }

    struct Model::Impl {
        std::vector<Tree> trees;
        size_t feature_count = 0;

        Impl(const nlohmann::json& model) {
            const auto& ts = model.at("oblivious_trees");

            for (const auto& t : ts) {
                trees.emplace_back(t);
            }
            feature_count = model.at("features_info").at("float_features").size();
        }

        double predict(const float* f) const noexcept {
            double res = 0.0;
            for (const auto& t : trees) {
                res += t.calc(f);
            }

            return res;
        }

        void predict4(const float* f0, const float* f1, const float* f2, const float* f3, double *y) const noexcept {
            y[0] = 0.0;
            y[1] = 0.0;
            y[2] = 0.0;
            y[3] = 0.0;

            for (const auto& t : trees) {
                t.calc4(f0, f1, f2, f3, y);
            }
        }
    };

#else // NOSSE

// SSE4.1
#include <smmintrin.h>

    inline double Tree::calc(const float* f) const noexcept {
        size_t i = 0;
        __m128i one = _mm_set_epi32(8, 4, 2, 1);
        __m128i idx = _mm_set_epi32(0, 0, 0, 0);

        for (; i + 3 < borders.size(); i += 4) {
            __m128 x = _mm_set_ps(f[indexes[i + 3]], f[indexes[i + 2]], f[indexes[i + 1]], f[indexes[i]]);
            __m128 b = _mm_load_ps(borders.data() + i);
            __m128i cmp = _mm_castps_si128(_mm_cmpgt_ps(x, b));

            idx = _mm_or_si128(idx, _mm_and_si128(one, cmp));
            one = _mm_slli_epi32(one, 4);
        }

        // Combine indexes to one index
        idx = _mm_or_si128(idx, _mm_srli_si128(idx, 8));
        idx = _mm_or_si128(idx, _mm_srli_si128(idx, 4));

        uint32_t sidx = _mm_cvtsi128_si32(idx);
        uint32_t sone = _mm_cvtsi128_si32(one);

        for (; i < borders.size(); ++i) {
            if (f[indexes[i]] >= borders[i]) {
                sidx |= sone;
            }
            sone <<= 1;
        }

        return values[sidx];
    }

    inline void Tree::calc4(const float* f0, const float* f1, const float* f2, const float* f3, double *res) const noexcept {
        // Basically this is the same code as the single one but made in parallel for 4 examples.
        __m128i one = _mm_set_epi32(1, 1, 1, 1);
        __m128i idx = _mm_set_epi32(0, 0, 0, 0);

        for (size_t i = 0; i < borders.size(); ++i) {
            size_t cur_idx = indexes[i];
            __m128 x = _mm_set_ps(f0[cur_idx], f1[cur_idx], f2[cur_idx], f3[cur_idx]);
            __m128 b = _mm_set_ps(borders[i], borders[i], borders[i], borders[i]);
            __m128i cmp = _mm_castps_si128(_mm_cmpgt_ps(x, b));

            idx = _mm_or_si128(idx, _mm_and_si128(one, cmp));
            one = _mm_slli_epi32(one, 1);
        }

        alignas(16) uint32_t iii[4];
        _mm_store_si128((__m128i*)iii, idx); //_mm_castsi128_ps(idx));

        res[0] += values[iii[3]];
        res[1] += values[iii[2]];
        res[2] += values[iii[1]];
        res[3] += values[iii[0]];
    }

    struct Model::Impl {

        // 4 trees combined. It allows us to use SSE for parallel comparations.
        struct Tree4 {
            std::vector<double> values;
            std::vector<float> borders;
            std::vector<uint32_t> indexes;

            Tree4(const Tree& t1, const Tree& t2, const Tree& t3, const Tree& t4) {
                assert(t1.depth() == t2.depth() && t1.depth() == t3.depth() && t1.depth() == t4.depth());
                values.reserve(t1.values.size() * 4);
                values.insert(values.end(), t1.values.begin(), t1.values.end());
                values.insert(values.end(), t2.values.begin(), t2.values.end());
                values.insert(values.end(), t3.values.begin(), t3.values.end());
                values.insert(values.end(), t4.values.begin(), t4.values.end());

                for (size_t i = 0; i < t1.depth(); i++) {
                    borders.push_back(t1.borders[i]);
                    borders.push_back(t2.borders[i]);
                    borders.push_back(t3.borders[i]);
                    borders.push_back(t4.borders[i]);
                    indexes.push_back(t1.indexes[i]);
                    indexes.push_back(t2.indexes[i]);
                    indexes.push_back(t3.indexes[i]);
                    indexes.push_back(t4.indexes[i]);
                }
            }

            __m128d calc(const float* f) const noexcept {
                size_t offset = values.size() / 4;
                __m128i one = _mm_set_epi32(1, 1, 1, 1);
                __m128i idx = _mm_set_epi32(offset * 3, offset * 2, offset, 0);

                for (size_t i = 0; i < borders.size(); i += 4) {
                    __m128 x = _mm_set_ps(f[indexes[i + 3]], f[indexes[i + 2]], f[indexes[i + 1]], f[indexes[i]]);
                    __m128 b = _mm_load_ps(borders.data() + i);
                    __m128i cmp = _mm_castps_si128(_mm_cmpgt_ps(x, b));

                    idx = _mm_or_si128(idx, _mm_and_si128(one, cmp));
                    one = _mm_slli_epi32(one, 1);
                }

                alignas(16) uint32_t iii[4];
                _mm_store_si128((__m128i*)iii, idx); //_mm_castsi128_ps(idx));

                return _mm_set_pd(values[iii[0]] + values[iii[1]], values[iii[2]] + values[iii[3]]);
            }

            void calc4(const float* f0, const float* f1, const float* f2, const float* f3, double *res) const noexcept {
                size_t offset = values.size() / 4;
                __m128i one = _mm_set_epi32(1, 1, 1, 1);
                __m128i idx0 = _mm_set_epi32(offset * 3, offset * 2, offset, 0);
                __m128i idx1 = _mm_set_epi32(offset * 3, offset * 2, offset, 0);
                __m128i idx2 = _mm_set_epi32(offset * 3, offset * 2, offset, 0);
                __m128i idx3 = _mm_set_epi32(offset * 3, offset * 2, offset, 0);

                for (size_t i = 0; i < borders.size(); i += 4) {
                    __m128 b = _mm_load_ps(borders.data() + i);
                    __m128 x = _mm_set_ps(f0[indexes[i + 3]], f0[indexes[i + 2]], f0[indexes[i + 1]], f0[indexes[i]]);
                    __m128i cmp = _mm_castps_si128(_mm_cmpgt_ps(x, b));

                    // I start loading next value before doing index computation for parallelezation
                    x = _mm_set_ps(f1[indexes[i + 3]], f1[indexes[i + 2]], f1[indexes[i + 1]], f1[indexes[i]]);
                    idx0 = _mm_or_si128(idx0, _mm_and_si128(one, cmp));
                    cmp = _mm_castps_si128(_mm_cmpgt_ps(x, b));

                    x = _mm_set_ps(f2[indexes[i + 3]], f2[indexes[i + 2]], f2[indexes[i + 1]], f2[indexes[i]]);
                    idx1 = _mm_or_si128(idx1, _mm_and_si128(one, cmp));
                    cmp = _mm_castps_si128(_mm_cmpgt_ps(x, b));

                    x = _mm_set_ps(f3[indexes[i + 3]], f3[indexes[i + 2]], f3[indexes[i + 1]], f3[indexes[i]]);
                    idx2 = _mm_or_si128(idx2, _mm_and_si128(one, cmp));
                    cmp = _mm_castps_si128(_mm_cmpgt_ps(x, b));
                    idx3 = _mm_or_si128(idx3, _mm_and_si128(one, cmp));

                    one = _mm_slli_epi32(one, 1);
                }

                alignas(16) uint32_t iii[4];

                _mm_store_si128((__m128i*)iii, idx0);
                res[0] += values[iii[0]] + values[iii[1]] + values[iii[2]] + values[iii[3]];
                _mm_store_si128((__m128i*)iii, idx1);
                res[1] += values[iii[0]] + values[iii[1]] + values[iii[2]] + values[iii[3]];
                _mm_store_si128((__m128i*)iii, idx2);
                res[2] += values[iii[0]] + values[iii[1]] + values[iii[2]] + values[iii[3]];
                _mm_store_si128((__m128i*)iii, idx3);
                res[3] += values[iii[0]] + values[iii[1]] + values[iii[2]] + values[iii[3]];
            }
        };

        std::vector<Tree> trees;
        std::vector<Tree4> trees4;
        size_t feature_count = 0;

        Impl(const nlohmann::json& model) {
            const auto& ts = model.at("oblivious_trees");
            std::unordered_map<size_t, std::vector<Tree>> tmp;
            for (const auto& t : ts) {
                Tree xt{t};
                tmp[xt.depth()].emplace_back(std::move(xt));
            }

            // Now lets combine them:
            for (const auto& bucket : tmp) {
                const auto& vec = bucket.second;
                size_t i = 0;

                for (; i + 3 < vec.size(); i += 4) {
                    trees4.emplace_back(vec[i], vec[i + 1], vec[i + 2], vec[i + 3]);
                }

                for (; i < vec.size(); ++i)
                    trees.emplace_back(vec[i]);
            }

            feature_count = model.at("features_info").at("float_features").size();
        }

        double predict(const float* f) const noexcept {
            // Process 4-trees:
            __m128d r4 = _mm_set_pd(0.0, 0.0);
            for (const auto& t : trees4) {
                r4 = _mm_add_pd(r4, t.calc(f));
            }
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, r4);

            double res = tmp[0] + tmp[1];
            for (const auto& t : trees) {
                res += t.calc(f);
            }

            return res;
        }

        void predict4(const float* f0, const float* f1, const float* f2, const float* f3, double *y) const noexcept {
            y[0] = 0.0;
            y[1] = 0.0;
            y[2] = 0.0;
            y[3] = 0.0;

            // Process 4-trees:
            for (const auto& t : trees4) {
                t.calc4(f0, f1, f2, f3, y);
            }

            // Process single trees:
            for (const auto& t : trees) {
                t.calc4(f0, f1, f2, f3, y);
            }
        }
    };

#endif // NOSSE

    Model::Model() {
    }

    Model::Model(const std::string& filename) {
        load(filename);
    }

    Model::Model(std::istream& in) {
        load(in);
    }

    Model::~Model() = default;

    void Model::load(const std::string& filename) {
        std::ifstream in{filename};

        if (!in.good()) {
            throw std::runtime_error("Can't open file with model");
        }

        load(in);
    }

    void Model::load(std::istream& in) {
        nlohmann::json model = nlohmann::json::parse(in);

        impl_.reset(new Impl(model));
        scale_ = 1.0;
        bias_ = 0.0;

        if (model.count("scale_and_bias")) {
            const auto& scale_and_bias = model.at("scale_and_bias");
            if (scale_and_bias.size() == 2) {
                scale_ = scale_and_bias.at(0).get<double>();
                bias_ = scale_and_bias.at(1).at(0).get<double>();
            }
        }
    }

    double Model::apply(const float* features, size_t count) const {
        if (!impl_.get()) {
            throw std::runtime_error("Model is not loaded");
        }

        if (count < impl_->feature_count) {
            throw std::runtime_error("Not enough features");
        }

        return scale_ * impl_->predict(features) + bias_;
    }

    void Model::apply(const float* const* features, size_t size, size_t count, double* y) const {
        if (!impl_.get()) {
            throw std::runtime_error("Model is not loaded");
        }

        if (count < impl_->feature_count) {
            throw std::runtime_error("Not enough features");
        }

        size_t i = 0;
        for (i = 0; i + 4 <= size; i += 4) {
            impl_->predict4(features[i], features[i + 1], features[i + 2], features[i + 3], y + i);
            y[i] = scale_ * y[i] + bias_;
            y[i + 1] = scale_ * y[i + 1] + bias_;
            y[i + 2] = scale_ * y[i + 2] + bias_;
            y[i + 3] = scale_ * y[i + 3] + bias_;
        }

        for (; i < size; ++i) {
            y[i] = apply(features[i], count);
        }
    }

    size_t Model::feature_count() const {
        if (impl_.get()) {
            return impl_->feature_count;
        } else {
            return 0;
        }
    }

}
