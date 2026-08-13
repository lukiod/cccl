//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

template <class Config, class Level>
__device__ void test_this_group_view(const Config& config, Level level)
{
  const auto g = cudax::make_this_group(level, config);

  // We want to test just 1 unit, early exit all other groups.
  if (g.rank(cuda::grid) > 0)
  {
    return;
  }

  // Test view over the whole group
  {
    cudax::group_view gv{g};

    assert(cuda::gpu_thread.is_part_of(gv));
    assert(gv.count(level) == 1);
    assert(gv.rank(level) == 0);

    static_assert(gv.static_count(g) == 1);
    assert(gv.count(g) == 1);
    assert(gv.rank(g) == 0);

    gv.sync();
  }

  // Test view over the whole group with the unit changed.
  {
    cudax::group_view gv{cuda::gpu_thread, g};

    assert(cuda::gpu_thread.is_part_of(gv));
    assert(gv.count(level) == cuda::gpu_thread.count(g));
    assert(gv.rank(level) == cuda::gpu_thread.rank(g));

    static_assert(gv.static_count(g) == 1);
    assert(gv.count(g) == 1);
    assert(gv.rank(g) == 0);

    gv.sync();
  }

  // Test view of half of the group.
  {
    const auto n = cuda::gpu_thread.count_as<unsigned>(g) / 2;

    cudax::group_view gv{cuda::gpu_thread, g, cudax::take{n}};

    assert(cuda::gpu_thread.is_part_of(gv) == cuda::gpu_thread.rank(g) < n);
    if (cuda::gpu_thread.is_part_of(gv))
    {
      assert(gv.count(level) == n);
      assert(gv.rank(level) == cuda::gpu_thread.rank(g));

      static_assert(gv.static_count(g) == 1);
      assert(gv.count(g) == 1);
      assert(gv.rank(g) == 0);
    }
    gv.sync();
  }

  // Test view that groups threads by 4.
  if (!cuda::std::is_same_v<Level, cuda::thread_level>)
  {
    cudax::group_view gv{cuda::gpu_thread, g, cudax::group_by{4}};

    assert(cuda::gpu_thread.is_part_of(gv));
    assert(cuda::gpu_thread.count(gv) == n);
    assert(cuda::gpu_thread.rank(gv) == cuda::gpu_thread.rank(g) % 4);

    static_assert(gv.static_count(g) == cuda::std::dynamic_extent);
    assert(gv.count(g) == cuda::gpu_thread.count(g) / 4);
    assert(gv.rank(g) == cuda::gpu_thread.rank(g) / 4);

    gv.sync();
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    test_this_group_view(config, cuda::gpu_thread);
    test_this_group_view(config, cuda::warp);
    test_this_group_view(config, cuda::block);
    test_this_group_view(config, cuda::cluster);
    test_this_group_view(config, cuda::grid);
  }
};

C2H_TEST("Group View", "[group_view]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
  cuda::launch(stream, config, TestKernel{});

  if (cuda::device_attributes::compute_capability(device) >= cuda::compute_capability{90})
  {
    const auto config_cluster = cuda::make_config(
      cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
    cuda::launch(stream, config_cluster, TestKernel{});
  }

  stream.sync();
}
