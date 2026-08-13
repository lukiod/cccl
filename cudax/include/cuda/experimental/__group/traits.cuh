//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH
#define _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Synchronizer, class _Unit, class _ParentGroup, class _MappingResult>
using __group_synchronizer_instance_t = decltype(::cuda::std::declval<_Synchronizer>().make_instance(
  ::cuda::std::declval<const _Unit&>(),
  ::cuda::std::declval<const _ParentGroup&>(),
  ::cuda::std::declval<const _MappingResult&>()));

template <class _Tp, class = void>
inline constexpr bool __is_spannable = false;
template <class _Tp>
inline constexpr bool
  __is_spannable<_Tp, ::cuda::std::void_t<decltype(::cuda::std::span(::cuda::std::declval<_Tp>()))>> = true;

template <class _Span>
using _SpanElementType = typename _Span::element_type;

template <class _Span>
using _SpanValueType = typename _Span::value_type;

template <class _Unit, class _ParentGroup>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto __get_initial_mapping_result(const _ParentGroup& __parent) noexcept
{
  using _ParentMappingResult = typename _ParentGroup::__mapping_result_type;
  using _MappingResult =
    ::cuda::experimental::__mapping_result<1,
                                           ::cuda::experimental::__static_count_query_group<_Unit, _ParentGroup>(),
                                           _ParentMappingResult::is_always_exhaustive(),
                                           _ParentMappingResult::is_always_contiguous()>;
  return _MappingResult{
    1,
    0,
    ::cuda::experimental::__count_query_group<unsigned, _Unit>(__parent),
    ::cuda::experimental::__rank_query_group<unsigned, _Unit>(__parent),
    __parent.__mapping_result().lane_mask()};
}

template <class _Unit, class _ParentGroup, class _Mapping>
using __group_mapping_result_t = decltype(::cuda::std::declval<const _Mapping&>().map(
  ::cuda::std::declval<const _Unit&>(),
  ::cuda::std::declval<const _ParentGroup&>(),
  ::cuda::experimental::__get_initial_mapping_result<_Unit>(::cuda::std::declval<const _ParentGroup&>())));
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH
