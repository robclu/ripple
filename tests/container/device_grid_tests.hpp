//==--- ripple/core/tests/container/device_grid_tests.hpp -- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_grid_tests.hpp
/// \brief This file defines tests for device side grid functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_DEVICE_GRID_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_DEVICE_GRID_TESTS_HPP

#include <ripple/core/boundary/fo_extrap_loader.hpp>
#include <ripple/core/container/grid.hpp>

// Defines the data type for the tests.
using real_t = float;
using int_t  = int;

TEST(container_grid, can_apply_pipeline_on_device_1d) {
  constexpr auto size_x = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g(topo, size_x);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = static_cast<real_t>(global_idx(ripple::dim_x));
    }
  );
  g.apply_pipeline(pipeline);

  for (auto i : ripple::range(g.size())) {
    EXPECT_EQ(*g(i), static_cast<real_t>(i));
  }
}

TEST(container_grid, can_apply_pipeline_on_device_2d) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g(topo, size_x, size_y);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = static_cast<real_t>(global_idx(ripple::dim_x))
          + static_cast<real_t>(global_idx(ripple::dim_y));
    }
  );
  g.apply_pipeline(pipeline);

  for (auto j : ripple::range(g.size(ripple::dim_y))) {
    for (auto i : ripple::range(g.size(ripple::dim_x))) {
      EXPECT_EQ(*g(i, j), static_cast<real_t>(i + j));
    }
  }
}

TEST(container_grid, can_apply_pipeline_on_device_3d) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  constexpr auto size_z = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g(topo, size_x, size_y, size_z);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = static_cast<real_t>(global_idx(ripple::dim_x))
          + static_cast<real_t>(global_idx(ripple::dim_y))
          + static_cast<real_t>(global_idx(ripple::dim_z));
    }
  );
  g.apply_pipeline(pipeline);

  for (auto k : ripple::range(g.size(ripple::dim_z))) {
    for (auto j : ripple::range(g.size(ripple::dim_y))) {
      for (auto i : ripple::range(g.size(ripple::dim_x))) {
        EXPECT_EQ(*g(i, j, k), static_cast<real_t>(i + j + k));
      }
    }
  }
}

//==--- [two grid apply] ---------------------------------------------------==//

TEST(container_grid, can_apply_pipeline_on_device_1d_two_grid) {
  constexpr auto size_x = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g1(topo, size_x);
  ripple::grid_1d_t<int_t>  g2(topo, size_x);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it_1, auto it_2) {
      *it_1 = static_cast<real_t>(global_idx(ripple::dim_x));
      *it_2 = static_cast<int_t>(*it_1);
    }
  );
  g1.apply_pipeline(pipeline, g2);

  for (auto i : ripple::range(g1.size())) {
    EXPECT_EQ(*g1(i), static_cast<real_t>(i));
    EXPECT_EQ(*g2(i), static_cast<int_t>(i));
  }
}

TEST(container_grid, can_apply_pipeline_on_device_2d_two_grid) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g1(topo, size_x, size_y);
  ripple::grid_2d_t<int_t>  g2(topo, size_x, size_y);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it_1, auto it_2) {
      *it_1 = static_cast<real_t>(global_idx(ripple::dim_x))
            + static_cast<real_t>(global_idx(ripple::dim_y));
      *it_2 = static_cast<int_t>(*it_1);
    }
  );
  g1.apply_pipeline(pipeline, g2);

  for (auto j : ripple::range(g1.size(ripple::dim_y))) {
    for (auto i : ripple::range(g1.size(ripple::dim_x))) {
      const auto v = static_cast<real_t>(i + j);
      EXPECT_EQ(*g1(i, j), v);
      EXPECT_EQ(*g2(i, j), static_cast<int_t>(v));
    }
  }
}

TEST(container_grid, can_apply_pipeline_on_device_3d_two_grid) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  constexpr auto size_z = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g1(topo, size_x, size_y, size_z);
  ripple::grid_3d_t<int_t>  g2(topo, size_x, size_y, size_z);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it_1, auto it_2) {
      *it_1 = static_cast<real_t>(global_idx(ripple::dim_x))
            + static_cast<real_t>(global_idx(ripple::dim_y))
            + static_cast<real_t>(global_idx(ripple::dim_z));
      *it_2 = static_cast<int_t>(*it_1);
    }
  );
  g1.apply_pipeline(pipeline, g2);

  for (auto k : ripple::range(g1.size(ripple::dim_z))) {
    for (auto j : ripple::range(g1.size(ripple::dim_y))) {
      for (auto i : ripple::range(g1.size(ripple::dim_x))) {
        const auto v = static_cast<real_t>(i + j + k);
        EXPECT_EQ(*g1(i, j, k), v);
        EXPECT_EQ(*g2(i, j, k), static_cast<int_t>(v));
      }
    }
  }
}

//==--- [three grid apply] -------------------------------------------------==//

TEST(container_grid, can_apply_pipeline_on_device_1d_three_grid) {
  constexpr auto size_x = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g1(topo, size_x);
  ripple::grid_1d_t<int_t>  g2(topo, size_x);
  ripple::grid_1d_t<int_t>  g3(topo, size_x);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it_1, auto it_2, auto it_3) {
      *it_1 = static_cast<real_t>(global_idx(ripple::dim_x));
      *it_2 = static_cast<int_t>(*it_1);
      *it_3 = static_cast<int_t>(*it_2);
    }
  );
  g1.apply_pipeline(pipeline, g2, g3);

  for (auto i : ripple::range(g1.size())) {
    EXPECT_EQ(*g1(i), static_cast<real_t>(i));
    EXPECT_EQ(*g2(i), static_cast<int_t>(i));
    EXPECT_EQ(*g3(i), static_cast<int_t>(i));
  }
}

TEST(container_grid, can_apply_pipeline_on_device_2d_three_grid) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g1(topo, size_x, size_y);
  ripple::grid_2d_t<int_t>  g2(topo, size_x, size_y);
  ripple::grid_2d_t<int_t>  g3(topo, size_x, size_y);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it_1, auto it_2, auto it_3) {
      *it_1 = static_cast<real_t>(global_idx(ripple::dim_x))
            + static_cast<real_t>(global_idx(ripple::dim_y));
      *it_2 = static_cast<int_t>(*it_1);
      *it_3 = static_cast<int_t>(*it_2);
    }
  );
  g1.apply_pipeline(pipeline, g2, g3);

  for (auto j : ripple::range(g1.size(ripple::dim_y))) {
    for (auto i : ripple::range(g1.size(ripple::dim_x))) {
      const auto v = static_cast<real_t>(i + j);
      EXPECT_EQ(*g1(i, j), v);
      EXPECT_EQ(*g2(i, j), static_cast<int_t>(v));
      EXPECT_EQ(*g3(i, j), static_cast<int_t>(v));
    }
  }
}

TEST(container_grid, can_apply_pipeline_on_device_3d_three_grid) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  constexpr auto size_z = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g1(topo, size_x, size_y, size_z);
  ripple::grid_3d_t<int_t>  g2(topo, size_x, size_y, size_z);
  ripple::grid_3d_t<int_t>  g3(topo, size_x, size_y, size_z);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it_1, auto it_2, auto it_3) {
      *it_1 = static_cast<real_t>(global_idx(ripple::dim_x))
            + static_cast<real_t>(global_idx(ripple::dim_y))
            + static_cast<real_t>(global_idx(ripple::dim_z));
      *it_2 = static_cast<int_t>(*it_1);
      *it_3 = static_cast<int_t>(*it_2);
    }
  );
  g1.apply_pipeline(pipeline, g2, g3);

  for (auto k : ripple::range(g1.size(ripple::dim_z))) {
    for (auto j : ripple::range(g1.size(ripple::dim_y))) {
      for (auto i : ripple::range(g1.size(ripple::dim_x))) {
        const auto v = static_cast<real_t>(i + j + k);
        EXPECT_EQ(*g1(i, j, k), v);
        EXPECT_EQ(*g2(i, j, k), static_cast<int_t>(v));
        EXPECT_EQ(*g3(i, j, k), static_cast<int_t>(v));
      }
    }
  }
}

TEST(container_grid, can_apply_pipeline_on_device_3d_three_grid_non_shared) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  constexpr auto size_z = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g1(topo, size_x, size_y, size_z);
  ripple::grid_3d_t<int_t>  g2(topo, size_x, size_y, size_z);
  ripple::grid_3d_t<int_t>  g3(topo, size_x, size_y, size_z);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it_1, auto it_2, auto it_3) {
      *it_1 = static_cast<real_t>(global_idx(ripple::dim_x))
            + static_cast<real_t>(global_idx(ripple::dim_y))
            + static_cast<real_t>(global_idx(ripple::dim_z));
      *it_2 = static_cast<int_t>(*it_1);
      *it_3 = static_cast<int_t>(*it_2);
    }
  );
  g1.apply_pipeline_non_shared(pipeline, g2, g3);

  for (auto k : ripple::range(g1.size(ripple::dim_z))) {
    for (auto j : ripple::range(g1.size(ripple::dim_y))) {
      for (auto i : ripple::range(g1.size(ripple::dim_x))) {
        const auto v = static_cast<real_t>(i + j + k);
        EXPECT_EQ(*g1(i, j, k), v);
        EXPECT_EQ(*g2(i, j, k), static_cast<int_t>(v));
        EXPECT_EQ(*g3(i, j, k), static_cast<int_t>(v));
      }
    }
  }
}

//==--- [normalizes indices] -----------------------------------------------==//

TEST(container_grid, can_get_normalized_global_index_1d) {
  constexpr auto size_x = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g(topo, size_x);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = global_norm_idx(ripple::dim_x);
    }
  );
  g.apply_pipeline(pipeline);

  for (auto i : ripple::range(g.size())) {
    const auto norm_idx = (static_cast<real_t>(i) + 0.5) / size_x;
    EXPECT_NEAR(*g(i), norm_idx, 1e-4);
  }
}

TEST(container_grid, can_get_normalized_global_index_2d) {
  constexpr auto size_x = size_t{101};
  constexpr auto size_y = size_t{101};
  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g(topo, size_x, size_y);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = global_norm_idx(ripple::dim_x)
          + global_norm_idx(ripple::dim_y);
    }
  );
  g.apply_pipeline(pipeline);

  for (auto j : ripple::range(g.size(ripple::dim_y))) {
    for (auto i : ripple::range(g.size(ripple::dim_x))) {
      const auto norm_idx_x = (static_cast<real_t>(i) + 0.5) / size_x;
      const auto norm_idx_y = (static_cast<real_t>(j) + 0.5) / size_y;
      EXPECT_NEAR(*g(i, j), norm_idx_x + norm_idx_y, 1e-4);
    }
  }
}

TEST(container_grid, can_get_normalized_global_index_3d) {
  constexpr auto size_x = size_t{107};
  constexpr auto size_y = size_t{64};
  constexpr auto size_z = size_t{73};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g(topo, size_x, size_y, size_z);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = ripple::global_norm_idx(ripple::dim_x)
          + ripple::global_norm_idx(ripple::dim_y)
          + ripple::global_norm_idx(ripple::dim_z);
    }
  );
  g.apply_pipeline(pipeline);

  for (auto k : ripple::range(g.size(ripple::dim_z))) {
    for (auto j : ripple::range(g.size(ripple::dim_y))) {
      for (auto i : ripple::range(g.size(ripple::dim_x))) {
        const auto norm_idx_x = (static_cast<real_t>(i) + 0.5) / size_x;
        const auto norm_idx_y = (static_cast<real_t>(j) + 0.5) / size_y;
        const auto norm_idx_z = (static_cast<real_t>(k) + 0.5) / size_z;
        EXPECT_NEAR(*g(i, j, k), norm_idx_x + norm_idx_y + norm_idx_z, 1e-4);
      }
    }
  }
}

//==--- [boundary loading] -------------------------------------------------==//

TEST(container_grid, can_load_global_boundaries_3d) {
  constexpr auto size_x   = size_t{19};
  constexpr auto size_y   = size_t{51};
  constexpr auto size_z   = size_t{22};
  constexpr auto padding  = size_t{2};
  constexpr auto data_val = real_t{10};

  auto topo   = ripple::Topology();
  auto loader = ripple::FOExtrapLoader();

  ripple::grid_3d_t<real_t> g_in(topo, padding, size_x, size_y, size_z);
  ripple::grid_3d_t<real_t> g_out(topo, size_x, size_y, size_z);
    
  g_in.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) { *it = data_val; }
  ));
  g_in.load_boundaries(loader);

  constexpr auto cells_dim = padding * 2 + 1;
  auto sum_pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto in, auto out) {

      // Shift into the padding;
      constexpr auto dims = in.dimensions();
      ripple::unrolled_for<dims>([&] (auto d) {
        in.shift(d, -static_cast<int64_t>(padding));
      });
      auto sum = real_t{0};
 
      // Sum 3 x 3 cells:
      for (auto z : ripple::range(cells_dim)) {
        for (auto y : ripple::range(cells_dim)) {
          for (auto x : ripple::range(cells_dim)) {
            sum += 
               *(in.offset(ripple::dim_z, z)
                   .offset(ripple::dim_y, y)
                   .offset(ripple::dim_x, x));
          }
        }
      }
      *out = sum;
    }
  );
  g_in.apply_pipeline_non_shared(sum_pipeline, g_out);

  const auto cell_sum = cells_dim * cells_dim * cells_dim * data_val;
  for (auto k : ripple::range(g_out.size(ripple::dim_z))) {
    for (auto j : ripple::range(g_out.size(ripple::dim_y))) {
      for (auto i : ripple::range(g_out.size(ripple::dim_x))) {
        EXPECT_EQ(*g_out(i, j, k), cell_sum);
      }
    }
  } 
}

//==--- [reduction] --------------------------------------------------------==//

TEST(container_grid, can_reduce_1d_no_padding) {
  constexpr auto size_x   = size_t{1207};
  constexpr auto data_val = real_t{10};

  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g(topo, size_x);
    
  g.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) { *it = data_val; }
  ));
  auto v = g.reduce(ripple::SumReducer());

  EXPECT_EQ(v, size_x * data_val);
}

TEST(container_grid, can_reduce_1d_with_padding) {
  constexpr auto size_x   = size_t{1207};
  constexpr auto padding  = size_t{2};
  constexpr auto data_val = real_t{10};

  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g(topo, padding, size_x);
    
  g.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) { *it = data_val; }
  ));
  auto v = g.reduce(ripple::SumReducer());

  EXPECT_EQ(v, size_x * data_val);
}

TEST(container_grid, can_reduce_2d_no_padding) {
  constexpr auto size_x   = size_t{127};
  constexpr auto size_y   = size_t{243};
  constexpr auto data_val = real_t{10};

  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g(topo, size_x, size_y);
    
  g.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) { *it = data_val; }
  ));
  auto v = g.reduce(ripple::SumReducer());

  EXPECT_EQ(v, size_x * size_y * data_val);
}

TEST(container_grid, can_reduce_2d_with_padding) {
  constexpr auto size_x   = size_t{127};
  constexpr auto size_y   = size_t{243};
  constexpr auto padding  = size_t{1};
  constexpr auto data_val = real_t{10};

  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g(topo, padding, size_x, size_y);
    
  g.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) { *it = data_val; }
  ));
  auto v = g.reduce(ripple::SumReducer());

  EXPECT_EQ(v, size_x * size_y * data_val);
}

TEST(container_grid, can_reduce_3d_no_padding) {
  constexpr auto size_x   = size_t{187};
  constexpr auto size_y   = size_t{113};
  constexpr auto size_z   = size_t{61};
  constexpr auto data_val = real_t{10};

  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g(topo, size_x, size_y, size_z);
    
  g.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) { *it = data_val; }
  ));
  auto v = g.reduce(ripple::SumReducer());

  EXPECT_EQ(v, size_x * size_y * size_z * data_val);
}

TEST(container_grid, can_reduce_3d_with_padding) {
  constexpr auto size_x   = size_t{187};
  constexpr auto size_y   = size_t{113};
  constexpr auto size_z   = size_t{61};
  constexpr auto padding  = size_t{2};
  constexpr auto data_val = real_t{10};

  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g(topo, padding, size_x, size_y, size_z);
    
  g.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) { *it = data_val; }
  ));
  auto v = g.reduce(ripple::SumReducer());

  EXPECT_EQ(v, size_x * size_y * size_z * data_val);
}

//==--- [swap] -------------------------------------------------------------==//

TEST(container_grid, can_swap_grids) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  constexpr auto size_z = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g1(topo, size_x, size_y, size_z);
  ripple::grid_3d_t<real_t> g2(topo, size_x, size_y, size_z);
    
  g1.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = static_cast<real_t>(global_idx(ripple::dim_x))
          + static_cast<real_t>(global_idx(ripple::dim_y))
          + static_cast<real_t>(global_idx(ripple::dim_z));
    }
  ));

  g2.apply_pipeline(ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = static_cast<real_t>(global_idx(ripple::dim_x)) + 0.3f;
    }
  ));

  using std::swap;
  swap(g1, g2);

  for (auto k : ripple::range(g1.size(ripple::dim_z))) {
    for (auto j : ripple::range(g1.size(ripple::dim_y))) {
      for (auto i : ripple::range(g1.size(ripple::dim_x))) {
        const auto v = static_cast<real_t>(i + j + k);
        EXPECT_EQ(*g1(i, j, k), static_cast<real_t>(i) + 0.3f);
        EXPECT_EQ(*g2(i, j, k), v);
      }
    }
  }
}

#endif // RIPPLE_TESTS_CONTAINER_DEVICE_GRID_TESTS_HPP
