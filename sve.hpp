/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#pragma once

#include "simd_common.hpp"

#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>

namespace SIMD_NAMESPACE
{

namespace simd_abi
{

class sve
{
};

} // namespace simd_abi

template <> class simd_mask<float, simd_abi::sve>
{
    uint32_t m_value __attribute__((vector_size(512 / 8)));

public:
    using value_type = bool;
    using simd_type = simd<float, simd_abi::sve>;
    using abi_type = simd_abi::sve;
    SIMD_ALWAYS_INLINE inline simd_mask() = default;
    SIMD_ALWAYS_INLINE inline simd_mask(bool value)
    {
        m_value = svdup_n_u32_x(svptrue_b32(), value ? 1 : 0);
    }
    SIMD_ALWAYS_INLINE inline static constexpr int size()
    {
        return 512 / 8 / sizeof(float);
    }
    SIMD_ALWAYS_INLINE inline constexpr simd_mask(svbool_t const &value_in)
        : m_value(svdup_n_u32_z(value_in, 1))
    {
    }
    SIMD_ALWAYS_INLINE inline constexpr svbool_t get() const
    {
        return svcmpne_n_u32(svptrue_b32(), m_value, 0);
    }
    SIMD_ALWAYS_INLINE inline simd_mask operator||(simd_mask const &other) const
    {
        return simd_mask(svorr_z(svptrue_b32(), get(), other.get()));
    }
    SIMD_ALWAYS_INLINE inline simd_mask operator&&(simd_mask const &other) const
    {
        return simd_mask(svand_z(svptrue_b32(), get(), other.get()));
    }
    SIMD_ALWAYS_INLINE inline simd_mask operator!() const
    {
        return simd_mask(svnot_z(svptrue_b32(), get()));
    }
};

SIMD_ALWAYS_INLINE inline bool all_of(simd_mask<float, simd_abi::sve> const &a)
{
    return !svptest_any(svptrue_b32(), svnot_z(svptrue_b32(), a.get()));
}

SIMD_ALWAYS_INLINE inline bool any_of(simd_mask<float, simd_abi::sve> const &a)
{
    return svptest_any(a.get(), svptrue_b32());
}

template <> class simd<float, simd_abi::sve>
{
    float32_t m_value __attribute__((vector_size(512 / 8)));

public:
    using value_type = float;
    using abi_type = simd_abi::sve;
    using mask_type = simd_mask<float, abi_type>;
    using storage_type = simd_storage<float, abi_type>;
    SIMD_ALWAYS_INLINE inline simd() = default;
    SIMD_ALWAYS_INLINE inline static constexpr int size()
    {
        return 512 / 8 / sizeof(float);
    }
    SIMD_ALWAYS_INLINE inline simd(float value) : m_value(svdup_f32(value)) {}
    SIMD_ALWAYS_INLINE inline simd(storage_type const &value)
    {
        copy_from(value.data(), element_aligned_tag());
    }
    SIMD_ALWAYS_INLINE inline simd &operator=(storage_type const &value)
    {
        copy_from(value.data(), element_aligned_tag());
        return *this;
    }
    template <class Flags>
    SIMD_ALWAYS_INLINE inline simd(float const *ptr, Flags /*flags*/)
        : m_value(svld1_f32(svptrue_b32(), ptr))
    {
    }
    SIMD_ALWAYS_INLINE inline simd(float const *ptr, int stride)
    {
        svint32_t offsets = svindex_s32(0, stride);
        m_value = svld1_gather_s32offset_f32(svptrue_b32(), ptr, offsets);
    }
    SIMD_ALWAYS_INLINE inline constexpr simd(svfloat32_t const &value_in)
        : m_value(value_in)
    {
    }
    SIMD_ALWAYS_INLINE inline simd operator*(simd const &other) const
    {
        return simd(svmul_f32_x(svptrue_b32(), m_value, other.m_value));
    }
    SIMD_ALWAYS_INLINE inline simd operator/(simd const &other) const
    {
        return simd(svdiv_f32_x(svptrue_b32(), m_value, other.m_value));
    }
    SIMD_ALWAYS_INLINE inline simd operator+(simd const &other) const
    {
        return simd(svadd_f32_x(svptrue_b32(), m_value, other.m_value));
    }
    SIMD_ALWAYS_INLINE inline simd operator-(simd const &other) const
    {
        return simd(svsub_f32_x(svptrue_b32(), m_value, other.m_value));
    }
    SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd operator-() const
    {
        return simd(svneg_f32_x(svptrue_b32(), m_value));
    }
    SIMD_ALWAYS_INLINE inline void copy_from(float const *ptr,
                                             element_aligned_tag)
    {
        m_value = svld1_f32(svptrue_b32(), ptr);
    }
    SIMD_ALWAYS_INLINE inline void copy_to(float *ptr,
                                           element_aligned_tag) const
    {
        svst1_f32(svptrue_b32(), ptr, m_value);
    }
    SIMD_ALWAYS_INLINE inline constexpr svfloat32_t get() const
    {
        return m_value;
    }
    SIMD_ALWAYS_INLINE inline simd_mask<float, simd_abi::sve>
    operator<(simd const &other) const
    {
        return svcmplt_f32(svptrue_b32(), m_value, other.m_value);
    }
    SIMD_ALWAYS_INLINE inline simd_mask<float, simd_abi::sve>
    operator==(simd const &other) const
    {
        return svcmpeq_f32(svptrue_b32(), m_value, other.m_value);
    }
};

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sve>
multiplysign(simd<float, simd_abi::sve> const &a,
             simd<float, simd_abi::sve> const &b)
{
    svuint32_t const sign_mask = svdup_u32(0x7fffffff);

    return simd<float, simd_abi::sve>(svreinterpret_f32_u32(
        sveor_u32_x(svptrue_b32(), svreinterpret_u32_f32(a.get()),
                    svand_u32_x(svptrue_b32(), sign_mask,
                                svreinterpret_u32_f32(b.get())))));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sve>
copysign(simd<float, simd_abi::sve> const &a,
         simd<float, simd_abi::sve> const &b)
{
    svuint32_t const sign_mask = svdup_u32(0x7fffffff);

    return simd<float, simd_abi::sve>(svreinterpret_f32_u32(sveor_u32_x(
        svptrue_b32(),
        svbic_u32_x(svptrue_b32(), sign_mask, svreinterpret_u32_f32(a.get())),
        svand_u32_x(svptrue_b32(), sign_mask,
                    svreinterpret_u32_f32(b.get())))));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sve>
abs(simd<float, simd_abi::sve> const &a)
{
    return simd<float, simd_abi::sve>(svabs_f32_z(svptrue_b32(), a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sve>
sqrt(simd<float, simd_abi::sve> const &a)
{
    return simd<float, simd_abi::sve>(svsqrt_f32_z(svptrue_b32(), a.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sve>
fma(simd<float, simd_abi::sve> const &a, simd<float, simd_abi::sve> const &b,
    simd<float, simd_abi::sve> const &c)
{
    return simd<float, simd_abi::sve>(
        svmad_f32_x(svptrue_b32(), a.get(), b.get(), c.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sve>
max(simd<float, simd_abi::sve> const &a, simd<float, simd_abi::sve> const &b)
{
    return simd<float, simd_abi::sve>(
        svmax_f32_x(svptrue_b32(), a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sve>
min(simd<float, simd_abi::sve> const &a, simd<float, simd_abi::sve> const &b)
{
    return simd<float, simd_abi::sve>(
        svmin_f32_x(svptrue_b32(), a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<float, simd_abi::sve>
choose(simd_mask<float, simd_abi::sve> const &a,
       simd<float, simd_abi::sve> const &b, simd<float, simd_abi::sve> const &c)
{
    return simd<float, simd_abi::sve>(svsel_f32(a.get(), b.get(), c.get()));
}

template <> class simd_mask<double, simd_abi::sve>
{
    uint64_t m_value __attribute__((vector_size(512 / 8)));

public:
    using value_type = bool;
    using simd_type = simd<double, simd_abi::sve>;
    using abi_type = simd_abi::sve;
    SIMD_ALWAYS_INLINE inline simd_mask() = default;
    SIMD_ALWAYS_INLINE inline simd_mask(bool value)
    {
        m_value = svdup_n_u64_x(svptrue_b64(), value ? 1 : 0);
    }
    SIMD_ALWAYS_INLINE inline static constexpr int size()
    {
        return 512 / 8 / sizeof(double);
    }
    SIMD_ALWAYS_INLINE inline constexpr simd_mask(svbool_t const &value_in)
        : m_value(svdup_n_u64_z(value_in, 1))
    {
    }
    SIMD_ALWAYS_INLINE inline constexpr svbool_t get() const
    {
        return svcmpne_n_u64(svptrue_b64(), m_value, 0);
    }
    SIMD_ALWAYS_INLINE inline simd_mask operator||(simd_mask const &other) const
    {
        return simd_mask(svorr_z(svptrue_b64(), get(), other.get()));
    }
    SIMD_ALWAYS_INLINE inline simd_mask operator&&(simd_mask const &other) const
    {
        return simd_mask(svand_z(svptrue_b64(), get(), other.get()));
    }
    SIMD_ALWAYS_INLINE inline simd_mask operator!() const
    {
        return simd_mask(svnot_z(svptrue_b64(), get()));
    }
};

SIMD_ALWAYS_INLINE inline bool all_of(simd_mask<double, simd_abi::sve> const &a)
{
    return !svptest_any(svptrue_b64(), svnot_z(svptrue_b64(), a.get()));
}

SIMD_ALWAYS_INLINE inline bool any_of(simd_mask<double, simd_abi::sve> const &a)
{
    return svptest_any(a.get(), svptrue_b64());
}

template <> class simd<double, simd_abi::sve>
{
    float64_t m_value __attribute__((vector_size(512 / 8)));

public:
    using value_type = double;
    using abi_type = simd_abi::sve;
    using mask_type = simd_mask<double, abi_type>;
    using storage_type = simd_storage<double, abi_type>;
    SIMD_ALWAYS_INLINE inline simd() = default;
    SIMD_ALWAYS_INLINE inline static constexpr int size()
    {
        return 512 / 8 / sizeof(double);
    }
    SIMD_ALWAYS_INLINE inline simd(double value) : m_value(svdup_f64(value)) {}
    SIMD_ALWAYS_INLINE inline simd(storage_type const &value)
    {
        copy_from(value.data(), element_aligned_tag());
    }
    SIMD_ALWAYS_INLINE inline simd &operator=(storage_type const &value)
    {
        copy_from(value.data(), element_aligned_tag());
        return *this;
    }
    template <class Flags>
    SIMD_ALWAYS_INLINE inline simd(double const *ptr, Flags /*flags*/)
        : m_value(svld1_f64(svptrue_b64(), ptr))
    {
    }
    SIMD_ALWAYS_INLINE inline simd(double const *ptr, int stride)
    {
        svint64_t offsets = svindex_s64(0, stride);
        m_value = svld1_gather_s64offset_f64(svptrue_b32(), ptr, offsets);
    }
    SIMD_ALWAYS_INLINE inline constexpr simd(svfloat64_t const &value_in)
        : m_value(value_in)
    {
    }
    SIMD_ALWAYS_INLINE inline simd operator*(simd const &other) const
    {
        return simd(svmul_f64_x(svptrue_b64(), m_value, other.m_value));
    }
    SIMD_ALWAYS_INLINE inline simd operator/(simd const &other) const
    {
        return simd(svdiv_f64_x(svptrue_b64(), m_value, other.m_value));
    }
    SIMD_ALWAYS_INLINE inline simd operator+(simd const &other) const
    {
        return simd(svadd_f64_x(svptrue_b64(), m_value, other.m_value));
    }
    SIMD_ALWAYS_INLINE inline simd operator-(simd const &other) const
    {
        return simd(svsub_f64_x(svptrue_b64(), m_value, other.m_value));
    }
    SIMD_ALWAYS_INLINE SIMD_HOST_DEVICE inline simd operator-() const
    {
        return simd(svneg_f64_x(svptrue_b64(), m_value));
    }
    SIMD_ALWAYS_INLINE inline void copy_from(double const *ptr,
                                             element_aligned_tag)
    {
        m_value = svld1_f64(svptrue_b64(), ptr);
    }
    SIMD_ALWAYS_INLINE inline void copy_to(double *ptr,
                                           element_aligned_tag) const
    {
        svst1_f64(svptrue_b64(), ptr, m_value);
    }
    SIMD_ALWAYS_INLINE inline constexpr svfloat64_t get() const
    {
        return m_value;
    }
    SIMD_ALWAYS_INLINE inline simd_mask<double, simd_abi::sve>
    operator<(simd const &other) const
    {
        return svcmplt_f64(svptrue_b64(), m_value, other.m_value);
    }
    SIMD_ALWAYS_INLINE inline simd_mask<double, simd_abi::sve>
    operator==(simd const &other) const
    {
        return svcmpeq_f64(svptrue_b64(), m_value, other.m_value);
    }
};

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sve>
multiplysign(simd<double, simd_abi::sve> const &a,
             simd<double, simd_abi::sve> const &b)
{
    svuint64_t const sign_mask = svdup_u64(0x7fffffffffffffff);

    return simd<double, simd_abi::sve>(svreinterpret_f64_u64(
        sveor_u64_x(svptrue_b64(), svreinterpret_u64_f64(a.get()),
                    svand_u64_x(svptrue_b64(), sign_mask,
                                svreinterpret_u64_f64(b.get())))));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sve>
copysign(simd<double, simd_abi::sve> const &a,
         simd<double, simd_abi::sve> const &b)
{
    svuint64_t const sign_mask = svdup_u64(0x7fffffffffffffff);

    return simd<double, simd_abi::sve>(svreinterpret_f64_u64(sveor_u64_x(
        svptrue_b64(),
        svbic_u64_x(svptrue_b64(), sign_mask, svreinterpret_u64_f64(a.get())),
        svand_u64_x(svptrue_b64(), sign_mask,
                    svreinterpret_u64_f64(b.get())))));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sve>
abs(simd<double, simd_abi::sve> const &a)
{
    return simd<double, simd_abi::sve>(svabs_f64_z(svptrue_b64(), a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sve>
sqrt(simd<double, simd_abi::sve> const &a)
{
    return simd<double, simd_abi::sve>(svsqrt_f64_z(svptrue_b64(), a.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sve>
fma(simd<double, simd_abi::sve> const &a, simd<double, simd_abi::sve> const &b,
    simd<double, simd_abi::sve> const &c)
{
    return simd<double, simd_abi::sve>(
        svmad_f64_x(svptrue_b64(), a.get(), b.get(), c.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sve>
max(simd<double, simd_abi::sve> const &a, simd<double, simd_abi::sve> const &b)
{
    return simd<double, simd_abi::sve>(
        svmax_f64_x(svptrue_b64(), a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sve>
min(simd<double, simd_abi::sve> const &a, simd<double, simd_abi::sve> const &b)
{
    return simd<double, simd_abi::sve>(
        svmin_f64_x(svptrue_b64(), a.get(), b.get()));
}

SIMD_ALWAYS_INLINE inline simd<double, simd_abi::sve>
choose(simd_mask<double, simd_abi::sve> const &a,
       simd<double, simd_abi::sve> const &b,
       simd<double, simd_abi::sve> const &c)
{
    return simd<double, simd_abi::sve>(svsel_f64(a.get(), b.get(), c.get()));
}

} // namespace SIMD_NAMESPACE

#endif
