#pragma once

#ifndef NOSSE

// SSE4.1
#include <smmintrin.h>
#include <cstdint>

namespace catboost {

    struct Vec4i {
        __m128i v;
        Vec4i() : v(_mm_set_epi32(0, 0, 0, 0)) {}
        explicit Vec4i(__m128i x) : v(x) {}
        explicit Vec4i(__m128 x) : v(_mm_castps_si128(x)) {}
        explicit Vec4i(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3) : v(_mm_set_epi32(x0, x1, x2, x3)) { }
        explicit Vec4i(uint32_t x) : v(_mm_set_epi32(x, x, x, x)) { }
        Vec4i(const Vec4i&) = default;
        Vec4i(Vec4i&&) = default;
        Vec4i& operator=(const Vec4i&) = default;
        Vec4i& operator=(Vec4i&&) = default;
        ~Vec4i() = default;

        void load(const uint32_t* p) {
            v = _mm_load_si128(reinterpret_cast<const __m128i*>(p));
        }

        void loadu(const uint32_t* p) {
            v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        }

        void store(uint32_t* p) const {
            _mm_store_si128(reinterpret_cast<__m128i*>(p), v);
        }

        void storeu(uint32_t* p) const {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
        }

        Vec4i& operator+=(Vec4i x) {
            v = _mm_add_epi32(v, x.v);
            return *this;
        }

        Vec4i operator+(Vec4i x) const {
            return Vec4i(_mm_add_epi32(v, x.v));
        }

        Vec4i& operator-=(Vec4i x) {
            v = _mm_sub_epi32(v, x.v);
            return *this;
        }

        Vec4i operator-(Vec4i x) const {
            return Vec4i(_mm_sub_epi32(v, x.v));
        }

        Vec4i& operator*=(Vec4i x) {
            v = _mm_mul_epi32(v, x.v);
            return *this;
        }

        Vec4i operator*(Vec4i x) const {
            return Vec4i(_mm_mul_epi32(v, x.v));
        }

        Vec4i& operator&=(Vec4i x) {
            v = _mm_and_si128(v, x.v);
            return *this;
        }

        Vec4i operator&(Vec4i x) const {
            return Vec4i(_mm_and_si128(v, x.v));
        }

        Vec4i& operator|=(Vec4i x) {
            v = _mm_or_si128(v, x.v);
            return *this;
        }

        Vec4i operator|(Vec4i x) const {
            return Vec4i(_mm_or_si128(v, x.v));
        }

        Vec4i& operator^=(Vec4i x) {
            v = _mm_xor_si128(v, x.v);
            return *this;
        }

        Vec4i operator^(Vec4i x) const {
            return Vec4i(_mm_xor_si128(v, x.v));
        }

        Vec4i& operator<<=(uint32_t s) {
            v = _mm_slli_epi32(v, s);
            return *this;
        }

        Vec4i operator<<(uint32_t s) const {
            return Vec4i(_mm_slli_epi32(v, s));
        }

        Vec4i& operator>>=(uint32_t s) {
            v = _mm_srli_epi32(v, s);
            return *this;
        }

        Vec4i operator>>(uint32_t s) const {
            return Vec4i(_mm_srli_epi32(v, s));
        }

        uint32_t sum() const {
            // Let's allow optimizer to do it for us:
            alignas(16) uint32_t x[4];
            store(x);
            return x[0] + x[1] + x[2] + x[3];
        }

    };

    struct Vec4f {
        __m128 v;
        Vec4f() : v(_mm_setzero_ps()) { }
        explicit Vec4f(__m128 x) : v(x) { }
        explicit Vec4f(float x0, float x1, float x2, float x3) : v(_mm_set_ps(x0, x1, x2, x3)) {}
        explicit Vec4f(float x) : v(_mm_set_ps(x, x, x, x)) { }
        Vec4f(const Vec4f&) = default;
        Vec4f(Vec4f&&) = default;
        Vec4f& operator=(const Vec4f&) = default;
        Vec4f& operator=(Vec4f&&) = default;
        ~Vec4f() = default;

        void load(const float* f) {
            v = _mm_load_ps(f);
        }

        void loadu(const float* f) {
            v = _mm_loadu_ps(f);
        }

        void store(float* f) const {
            _mm_store_ps(f, v);
        }

        void storeu(float* f) const {
            _mm_storeu_ps(f, v);
        }

        Vec4i operator < (Vec4f x) {
            return Vec4i(_mm_cmplt_ps(v, x.v));
        }

        Vec4i operator <= (Vec4f x) {
            return Vec4i(_mm_cmple_ps(v, x.v));
        }

        Vec4i operator > (Vec4f x) {
            return Vec4i(_mm_cmpgt_ps(v, x.v));
        }

        Vec4i operator >= (Vec4f x) {
            return Vec4i(_mm_cmpge_ps(v, x.v));
        }

        Vec4i operator == (Vec4f x) {
            return Vec4i(_mm_cmpeq_ps(v, x.v));
        }

        Vec4i operator != (Vec4f x) {
            return Vec4i(_mm_cmpneq_ps(v, x.v));
        }

        Vec4f& operator+=(Vec4f x) {
            v = _mm_add_ps(v, x.v);
            return *this;
        }

        Vec4f operator+(Vec4f x) const {
            return Vec4f(_mm_add_ps(v, x.v));
        }

        Vec4f& operator-=(Vec4f x) {
            v = _mm_sub_ps(v, x.v);
            return *this;
        }

        Vec4f operator-(Vec4f x) const {
            return Vec4f(_mm_sub_ps(v, x.v));
        }

        Vec4f& operator*=(Vec4f x) {
            v = _mm_mul_ps(v, x.v);
            return *this;
        }

        Vec4f operator*(Vec4f x) const {
            return Vec4f(_mm_mul_ps(v, x.v));
        }

        Vec4f& operator/=(Vec4f x) {
            v = _mm_div_ps(v, x.v);
            return *this;
        }

        Vec4f operator/(Vec4f x) const {
            return Vec4f(_mm_div_ps(v, x.v));
        }
    };

} // namespace catboost

#endif
