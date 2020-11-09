#include "catboost.hpp"
#include "json.hpp"
#include <fstream>

namespace catboost {

    namespace {

        struct JsonTree {
            std::vector<double> values;
            std::vector<float> borders;
            std::vector<uint32_t> indexes;

            JsonTree(const nlohmann::json& t, size_t feature_count) {
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
                    if (indexes.back() >= feature_count) {
                        throw std::runtime_error("Invalid model: index is greater than feature count");
                    }
                }
            }

            size_t depth() const {
                return borders.size();
            }
        };

        struct JsonModel {
            size_t feature_count = 0;
            std::vector<JsonTree> trees;
            double bias = 0.0;
            double scale = 1.0;

            void load(const nlohmann::json& model) {
                feature_count = model.at("features_info").at("float_features").size();
                const auto& ts = model.at("oblivious_trees");
                scale = 1.0;
                bias = 0.0;

                for (const auto& t : ts) {
                    trees.emplace_back(t, feature_count);
                }

                if (model.count("scale_and_bias")) {
                    const auto& scale_and_bias = model.at("scale_and_bias");
                    if (scale_and_bias.size() == 2) {
                        scale = scale_and_bias.at(0).get<double>();
                        bias = scale_and_bias.at(1).at(0).get<double>();
                    }
                }

            }
        };

    } // anonymous namespace

#ifdef NOSSE

    struct Model::Impl {
        struct Split {
            float border = 0.0f;
            uint32_t index = 0;
            uint32_t count = 0;

            Split(float b, uint32_t i, uint32_t c)
                : border(b)
                , index(i)
                , count(c)
            {
            }

            uint32_t apply(const float* f, uint32_t one) const {
                if (f[index] > border)
                    return one;
                return 0;
            }
        };
        std::vector<Split> splits;
        std::vector<double> values;

        size_t feature_count = 0;

        Impl(const JsonModel& model) {
            feature_count = model.feature_count;
            splits.reserve(model.trees.size() * 6);
            values.reserve(model.trees.size() * 64);
            for (const auto& tree : model.trees) {
                for (size_t i = 0; i < tree.borders.size(); i++) {
                    splits.emplace_back(
                        tree.borders[i],
                        tree.indexes[i],
                        0
                    );
                }
                splits.back().count = tree.values.size();
                values.insert(values.end(), tree.values.begin(), tree.values.end());
            }
        }

        double predict(const float* f) const noexcept {
            double res = 0.0;
            uint32_t idx = 0;
            size_t off = 0;
            uint32_t one = 1;

            for (const auto& split : splits) {
                idx |= split.apply(f, one);
                one <<= 1;
                if (split.count) {
                    res += values[off + idx];
                    off += split.count;
                    one = 1;
                    idx = 0;
                }
            }

            return res;
        }

        void predict4(const float* f0, const float* f1, const float* f2, const float* f3, double *y) const noexcept {
            y[0] = 0.0;
            y[1] = 0.0;
            y[2] = 0.0;
            y[3] = 0.0;
            uint32_t idx0 = 0;
            uint32_t idx1 = 0;
            uint32_t idx2 = 0;
            uint32_t idx3 = 0;
            size_t off = 0;
            uint32_t one = 1;

            for (const auto& split : splits) {
                idx0 |= split.apply(f0, one);
                idx1 |= split.apply(f1, one);
                idx2 |= split.apply(f2, one);
                idx3 |= split.apply(f3, one);
                one <<= 1;
                if (split.count) {
                    y[0] += values[off + idx0];
                    y[1] += values[off + idx1];
                    y[2] += values[off + idx2];
                    y[3] += values[off + idx3];
                    off += split.count;
                    one = 1;
                    idx0 = 0;
                    idx1 = 0;
                    idx2 = 0;
                    idx3 = 0;
                }
            }
        }
    };

#else // NOSSE

// SSE4.1
#include "vec4.hpp"

    namespace {

        // Binary data reader/writer
        template <size_t align = 16>
        class Bin {
            std::vector<unsigned char> data_;

            static constexpr size_t aligned_size(size_t sz) {
                return (sz % align) == 0?
                    sz : align + sz - sz % align;
            }
        public:
            Bin() = default;
            ~Bin() = default;
            Bin(const Bin&) = delete;
            Bin(Bin&&) = delete;
            Bin&operator=(const Bin&) = delete;
            Bin&operator=(Bin&&) = delete;

            void write(const void* p, size_t sz) {
                size_t actual_size = aligned_size(sz);
                size_t delta = actual_size - sz;
                const unsigned char* pp = reinterpret_cast<const unsigned char*>(p);
                data_.insert(data_.end(), pp, pp + sz);
                for (size_t i = 0; i < delta; ++i)
                    data_.push_back(0);
            }

            template <typename T>
            void write(const T* x) {
                write(reinterpret_cast<const void*>(x), sizeof(T));
            }

            class Iterator {
                const unsigned char* pos = nullptr;
                const unsigned char* end = nullptr;

                friend class Bin;
                Iterator(const unsigned char* p, const unsigned char* e)
                    : pos(p), end(e)
                {
                }
            public:
                Iterator() = default;
                Iterator(const Iterator&) = default;
                Iterator(Iterator&&) = default;
                Iterator& operator=(const Iterator&) = default;
                Iterator& operator=(Iterator&&) = default;
                ~Iterator() = default;

                const void* read(size_t sz) {
                    size_t actual_size = aligned_size(sz);

                    if (pos >= end) {
                        return nullptr;
                    }

                    const void* res = pos;
                    pos += actual_size;

                    return res;
                }

                template <typename T>
                const T* read() {
                    return reinterpret_cast<const T*>(read(sizeof(T)));
                }
            };

            Iterator iter() const {
                return Iterator(data_.data(), data_.data() + data_.size());
            }
        };

    } // anonymous namespace

    struct Model::Impl {
        enum SplitType {
            SPLIT_SIMPLE,
            SPLIT4_MULTI_TREE,
            SPLIT4_SINGLE_TREE,
        };

        struct SplitInfo {
            uint32_t depth = 0;
            SplitType type = SPLIT_SIMPLE;
        };

        struct Split {
            float border = 0.0f;
            uint32_t index = 0;

            Split(float b, uint32_t i)
                : border(b)
                , index(i)
            {
            }

            uint32_t apply(const float* f, uint32_t one) const {
                if (f[index] > border)
                    return one;
                return 0;
            }
        };

        struct Split4 {
            Vec4f border{};
            uint32_t index[4] = { 0, 0, 0, 0 };

            Vec4i apply(const float* f, Vec4i one) const {
                Vec4f x{f[index[0]], f[index[1]], f[index[2]], f[index[3]]};
                return one & (x > border);
            }
        };

        Bin<16> splits;
        std::vector<double> values;

        void add_tree4(const JsonTree& t0, const JsonTree& t1, const JsonTree& t2, const JsonTree t3) {
            // Add meta info:
            SplitInfo info;
            info.depth = t0.depth();
            info.type = SPLIT4_MULTI_TREE;
            splits.write(&info);

            // Now add borders and indexes:
            for (size_t i = 0; i < t0.depth(); ++i) {
                Split4 s;
                s.border = Vec4f(t0.borders[i], t1.borders[i], t2.borders[i], t3.borders[i]);
                s.index[0] = t0.indexes[i];
                s.index[1] = t1.indexes[i];
                s.index[2] = t2.indexes[i];
                s.index[3] = t3.indexes[i];

                splits.write(&s);
            }

            values.insert(values.end(), t0.values.begin(), t0.values.end());
            values.insert(values.end(), t1.values.begin(), t1.values.end());
            values.insert(values.end(), t2.values.begin(), t2.values.end());
            values.insert(values.end(), t3.values.begin(), t3.values.end());
        }

        void add_tree(const JsonTree& t) {
            // Add meta info:
            SplitInfo info;
            info.depth = t.depth();
            info.type = SPLIT4_SINGLE_TREE;
            splits.write(&info);

            size_t i = 0;

            // Write optimized trees while we can
            for (i = 0; i + 4 <= t.depth(); i += 4) {
                Split4 s;
                s.border = Vec4f(t.borders[i + 3], t.borders[i + 2], t.borders[i + 1], t.borders[i]);
                s.index[0] = t.indexes[i + 3];
                s.index[1] = t.indexes[i + 2];
                s.index[2] = t.indexes[i + 1];
                s.index[3] = t.indexes[i + 0];
                splits.write(&s);
            }

            // Now write the rest:
            for (; i < t.depth(); ++i) {
                Split s{t.borders[i], t.indexes[i]};
                splits.write(&s);
            }

            values.insert(values.end(), t.values.begin(), t.values.end());
        }

        size_t feature_count = 0;

        Impl(const JsonModel& model) {
            feature_count = model.feature_count;

            std::unordered_map<size_t, std::vector<JsonTree>> tmp;
            for (const auto& t : model.trees) {
                JsonTree xt{t};
                tmp[xt.depth()].emplace_back(std::move(xt));
            }

            // Now lets combine them:
            for (const auto& bucket : tmp) {
                const auto& vec = bucket.second;
                size_t i = 0;

                for (; i + 3 < vec.size(); i += 4) {
                    add_tree4(vec[i], vec[i + 1], vec[i + 2], vec[i + 3]);
                }

                for (; i < vec.size(); ++i)
                    add_tree(vec[i]);
            }

        }

        double predict(const float* f) const noexcept {
            auto iter = splits.iter();
            double res = 0.0;
            uint32_t offset = 0;

            for (const SplitInfo* info = iter.read<SplitInfo>(); info != nullptr; info = iter.read<SplitInfo>()) {
                switch (info->type) {
                case SPLIT_SIMPLE:
                {
                    uint32_t one = 1;
                    uint32_t idx = 0;
                    for (uint32_t i = 0; i < info->depth; ++i) {
                        const Split* split = iter.read<Split>();
                        idx |= split->apply(f, one);
                        one <<= 1;
                    }

                    res += values[offset + idx];
                    offset += static_cast<uint32_t>(1) << info->depth;
                }
                break;

                case SPLIT4_SINGLE_TREE:
                {
                    uint32_t i = 0;
                    Vec4i one4{ 8, 4, 2, 1 };
                    Vec4i idx4{};

                    for (; i + 4 <= info->depth; i += 4) {
                        const Split4* split = iter.read<Split4>();
                        idx4 |= split->apply(f, one4);
                        one4 <<= 4;
                    }

                    uint32_t idx = idx4.sum();
                    uint32_t one = static_cast<uint32_t>(1) << i;

                    for (; i < info->depth; ++i) {
                        const Split* split = iter.read<Split>();
                        idx |= split->apply(f, one);
                        one <<= 1;
                    }

                    res += values[offset + idx];
                    offset += static_cast<uint32_t>(1) << info->depth;
                }
                break;

                case SPLIT4_MULTI_TREE:
                {
                    Vec4i idx{};
                    Vec4i one{1, 1, 1, 1};

                    for (uint32_t i = 0; i < info->depth; ++i) {
                        const Split4* split = iter.read<Split4>();
                        idx |= split->apply(f, one);
                        one <<= 1;
                    }

                    alignas(16) uint32_t index[4];
                    idx.store(index);

                    res += values[offset + index[3]];
                    offset += static_cast<uint32_t>(1) << info->depth;
                    res += values[offset + index[2]];
                    offset += static_cast<uint32_t>(1) << info->depth;
                    res += values[offset + index[1]];
                    offset += static_cast<uint32_t>(1) << info->depth;
                    res += values[offset + index[0]];
                    offset += static_cast<uint32_t>(1) << info->depth;
                }
                break;
                } // switch (info->type)
            }

            return res;
        }

        void predict4(const float* f0, const float* f1, const float* f2, const float* f3, double *y) const noexcept {
            y[0] = 0.0;
            y[1] = 0.0;
            y[2] = 0.0;
            y[3] = 0.0;

            auto iter = splits.iter();
            uint32_t offset = 0;

            for (const SplitInfo* info = iter.read<SplitInfo>(); info != nullptr; info = iter.read<SplitInfo>()) {
                switch (info->type) {
                case SPLIT_SIMPLE:
                { // This situation is impossible, but it could be used for debugging sometime.
                    uint32_t one = 1;
                    uint32_t idx0 = 0;
                    uint32_t idx1 = 0;
                    uint32_t idx2 = 0;
                    uint32_t idx3 = 0;
                    for (uint32_t i = 0; i < info->depth; ++i) {
                        const Split* split = iter.read<Split>();
                        idx0 |= split->apply(f0, one);
                        idx1 |= split->apply(f1, one);
                        idx2 |= split->apply(f2, one);
                        idx3 |= split->apply(f3, one);
                        one <<= 1;
                    }

                    y[0] += values[offset + idx0];
                    y[1] += values[offset + idx1];
                    y[2] += values[offset + idx2];
                    y[3] += values[offset + idx3];
                    offset += static_cast<uint32_t>(1) << info->depth;
                }
                break;

                case SPLIT4_SINGLE_TREE:
                {
                    uint32_t i = 0;
                    Vec4i one4{ 8, 4, 2, 1 };
                    Vec4i idx40{};
                    Vec4i idx41{};
                    Vec4i idx42{};
                    Vec4i idx43{};

                    for (; i + 4 <= info->depth; i += 4) {
                        const Split4* split = iter.read<Split4>();
                        idx40 |= split->apply(f0, one4);
                        idx41 |= split->apply(f1, one4);
                        idx42 |= split->apply(f2, one4);
                        idx43 |= split->apply(f3, one4);
                        one4 <<= 4;
                    }

                    uint32_t idx0 = idx40.sum();
                    uint32_t idx1 = idx41.sum();
                    uint32_t idx2 = idx42.sum();
                    uint32_t idx3 = idx43.sum();
                    uint32_t one = static_cast<uint32_t>(1) << i;

                    for (; i < info->depth; ++i) {
                        const Split* split = iter.read<Split>();
                        idx0 |= split->apply(f0, one);
                        idx1 |= split->apply(f1, one);
                        idx2 |= split->apply(f2, one);
                        idx3 |= split->apply(f3, one);
                        one <<= 1;
                    }

                    y[0] += values[offset + idx0];
                    y[1] += values[offset + idx1];
                    y[2] += values[offset + idx2];
                    y[3] += values[offset + idx3];
                    offset += static_cast<uint32_t>(1) << info->depth;
                }
                break;

                case SPLIT4_MULTI_TREE:
                {
                    Vec4i idx0{};
                    Vec4i idx1{};
                    Vec4i idx2{};
                    Vec4i idx3{};
                    Vec4i one{1, 1, 1, 1};

                    for (uint32_t i = 0; i < info->depth; ++i) {
                        const Split4* split = iter.read<Split4>();
                        idx0 |= split->apply(f0, one);
                        idx1 |= split->apply(f1, one);
                        idx2 |= split->apply(f2, one);
                        idx3 |= split->apply(f3, one);
                        one <<= 1;
                    }

                    alignas(16) uint32_t index0[4];
                    alignas(16) uint32_t index1[4];
                    alignas(16) uint32_t index2[4];
                    alignas(16) uint32_t index3[4];

                    idx0.store(index0);
                    idx1.store(index1);
                    idx2.store(index2);
                    idx3.store(index3);

                    y[0] += values[offset + index0[3]];
                    y[1] += values[offset + index1[3]];
                    y[2] += values[offset + index2[3]];
                    y[3] += values[offset + index3[3]];
                    offset += static_cast<uint32_t>(1) << info->depth;
                    y[0] += values[offset + index0[2]];
                    y[1] += values[offset + index1[2]];
                    y[2] += values[offset + index2[2]];
                    y[3] += values[offset + index3[2]];
                    offset += static_cast<uint32_t>(1) << info->depth;
                    y[0] += values[offset + index0[1]];
                    y[1] += values[offset + index1[1]];
                    y[2] += values[offset + index2[1]];
                    y[3] += values[offset + index3[1]];
                    offset += static_cast<uint32_t>(1) << info->depth;
                    y[0] += values[offset + index0[0]];
                    y[1] += values[offset + index1[0]];
                    y[2] += values[offset + index2[0]];
                    y[3] += values[offset + index3[0]];
                    offset += static_cast<uint32_t>(1) << info->depth;
                }
                break;
                } // switch (info->type)
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

        JsonModel jmodel;
        jmodel.load(model);

        impl_.reset(new Impl(jmodel));
        scale_ = jmodel.scale;
        bias_ = jmodel.bias;
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

} // namespace catboost
